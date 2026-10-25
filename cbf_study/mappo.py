"""
Compact MAPPO (PyTorch). Formation training runs WITHOUT the filter.

* Actor: one network shared by all agents (the formation observation carries the agent id).
  Gaussian policy whose mean is center + half * tanh(net(obs)), so the deterministic action (the
  mean) always lies in the action box; a state-independent log-std. The actor sees only its own
  agent's observation row (decentralised execution).
* Sampled actions are NOT squashed: the log-probability is that of the raw Gaussian sample, and
  the environment clips the sample to the box before applying it (exactly as the protocol clips
  commands). The clip is part of the environment, so the same raw sample and the same
  log-probability are used in the rollout and in the PPO ratio; the ratio is exactly 1 before the
  first gradient step of an update (checked every update and logged as ratio0_maxdev).
* Critic: centralised, V(concat of all agents' observations) for the shared team reward.
* PPO: GAE(gamma, lambda), clipped surrogate, advantage normalisation, running value
  normalisation, separate Adam optimisers, gradient-norm clipping.
* Termination vs time limit: a TERMINATED step (goal, collision) has no bootstrap; a TRUNCATED
  step (time limit) gets r += gamma * V(final state before reset) and then cuts the trace.

MappoLearner is environment-agnostic: any env with reset() -> obs [E,N,D], step(raw [E,N,A]) ->
StepOut(obs, reward [E], terminated [E], truncated [E], final_obs, episodes), n_envs, lo, hi.
train() wraps it for the formation task (FormationTrainEnv); scripts/validate_mappo_vmas.py runs it
on a stock VMAS scenario.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import StudyConfig
from .observations import obs_dim as _obs_dim
from .protocol import Outcome
from .train_env import FormationTrainEnv

CKPT_FORMAT = 1


def mlp(n_in, hidden, n_out, out_gain):
    layers, d = [], n_in
    for h in hidden:
        lin = nn.Linear(d, h)
        nn.init.orthogonal_(lin.weight, np.sqrt(2)); nn.init.zeros_(lin.bias)
        layers += [lin, nn.Tanh()]
        d = h
    out = nn.Linear(d, n_out)
    nn.init.orthogonal_(out.weight, out_gain); nn.init.zeros_(out.bias)
    return nn.Sequential(*layers, out)


class Actor(nn.Module):
    def __init__(self, obs_dim, hidden, lo, hi, init_std_frac):
        super().__init__()
        lo = torch.as_tensor(lo, dtype=torch.float32)
        hi = torch.as_tensor(hi, dtype=torch.float32)
        self.body = mlp(obs_dim, hidden, len(lo), 0.01)
        self.register_buffer("a_center", (hi + lo) / 2)
        self.register_buffer("a_half", (hi - lo) / 2)
        self.log_std = nn.Parameter(torch.full((len(lo),), float(np.log(init_std_frac))))

    def mean(self, obs):
        return self.a_center + self.a_half * torch.tanh(self.body(obs))

    def std(self):
        return self.a_half * self.log_std.exp()

    def log_prob(self, obs, act):
        return torch.distributions.Normal(self.mean(obs), self.std()).log_prob(act).sum(-1)


class Critic(nn.Module):
    def __init__(self, joint_dim, hidden):
        super().__init__()
        self.net = mlp(joint_dim, hidden, 1, 1.0)

    def forward(self, joint):
        return self.net(joint).squeeze(-1)


class ValueNorm:
    """Running mean / variance of value targets (float64)."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, x):
        x = np.asarray(x, dtype=np.float64).ravel()
        n_b, mean_b, var_b = x.size, x.mean(), x.var()
        n = self.n + n_b
        delta = mean_b - self.mean
        self.mean += delta * n_b / n
        self.m2 += var_b * n_b + delta ** 2 * self.n * n_b / n
        self.n = n

    @property
    def std(self):
        return float(np.sqrt(self.m2 / self.n)) if self.n > 1 else 1.0

    def norm(self, x):
        return (x - self.mean) / max(self.std, 1e-6)

    def denorm(self, x):
        return x * max(self.std, 1e-6) + self.mean

    def state(self):
        return dict(n=self.n, mean=self.mean, m2=self.m2)

    def load(self, s):
        self.n, self.mean, self.m2 = s["n"], s["mean"], s["m2"]


def build_actor(meta):
    return Actor(meta["obs_dim"], meta["hidden_actor"], meta["lo"], meta["hi"], meta["init_std_frac"])


def save_actor(path, actor, meta):
    torch.save(dict(format=CKPT_FORMAT, actor_state=actor.state_dict(), **meta), path)


def load_actor_checkpoint(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != CKPT_FORMAT:
        raise ValueError(f"{path}: unknown checkpoint format {ck.get('format')}")
    actor = build_actor(ck)
    actor.load_state_dict(ck["actor_state"])
    return actor, ck


def _gae(rew, val, done, last_val, gamma, lam):
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    nxt_adv = np.zeros_like(last_val)
    nxt_val = last_val
    for t in reversed(range(T)):
        nd = 1.0 - done[t]
        delta = rew[t] + gamma * nxt_val * nd - val[t]
        nxt_adv = delta + gamma * lam * nd * nxt_adv
        adv[t] = nxt_adv
        nxt_val = val[t]
    return adv


class MappoLearner:
    def __init__(self, obs_dim, n_agents, lo, hi, hp, seed):
        self.hp = dict(hp)
        self.D, self.N = int(obs_dim), int(n_agents)
        torch.manual_seed(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.actor = Actor(self.D, list(hp["hidden_actor"]), lo, hi, float(hp["init_std_frac"]))
        self.critic = Critic(self.N * self.D, list(hp["hidden_critic"]))
        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=float(hp["lr_actor"]), eps=1e-5)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=float(hp["lr_critic"]), eps=1e-5)
        self.vnorm = ValueNorm()

    def set_hp(self, hp):
        self.hp = dict(hp)
        for opt, key in ((self.opt_a, "lr_actor"), (self.opt_c, "lr_critic")):
            for pg in opt.param_groups:
                pg["lr"] = float(hp[key])

    def value(self, obs):
        with torch.no_grad():
            return self.vnorm.denorm(self.critic(torch.from_numpy(obs.reshape(obs.shape[0], -1))).double().numpy())

    def sample(self, obs):
        """raw Gaussian sample (not clipped) and its log-probability."""
        with torch.no_grad():
            mu = self.actor.mean(torch.from_numpy(obs))
            std = self.actor.std()
            a = mu + std * torch.randn(mu.shape, generator=self.gen)
            logp = torch.distributions.Normal(mu, std).log_prob(a).sum(-1)
        return a.numpy(), logp.numpy()

    def collect(self, env, obs, T):
        gamma = float(self.hp["gamma"])
        E, N, D = obs.shape
        A = len(self.actor.a_half)
        b = dict(obs=np.zeros((T, E, N, D), np.float32), act=np.zeros((T, E, N, A), np.float32),
                 logp=np.zeros((T, E, N), np.float32), rew=np.zeros((T, E)), done=np.zeros((T, E)),
                 val=np.zeros((T, E)), term=np.zeros((T, E), bool), trunc=np.zeros((T, E), bool))
        episodes = []
        for t in range(T):
            a, logp = self.sample(obs)
            v = self.value(obs)
            so = env.step(a)
            trunc = so.truncated & ~so.terminated
            r = so.reward.astype(np.float64).copy()
            if trunc.any():
                r = np.where(trunc, r + gamma * self.value(so.final_obs), r)
            b["obs"][t], b["act"][t], b["logp"][t] = obs, a, logp
            b["rew"][t], b["done"][t], b["val"][t] = r, (so.terminated | trunc), v
            b["term"][t], b["trunc"][t] = so.terminated, trunc
            episodes += so.episodes
            obs = so.obs
        b["last_val"] = self.value(obs)
        return b, obs, episodes

    def update(self, b):
        hp = self.hp
        T, E, N, D = b["obs"].shape
        gamma, lam, clip = float(hp["gamma"]), float(hp["gae_lambda"]), float(hp["clip"])
        adv = _gae(b["rew"], b["val"], b["done"], b["last_val"], gamma, lam)
        ret = adv + b["val"]
        self.vnorm.update(ret)
        f_obs = torch.from_numpy(b["obs"].reshape(T * E, N, D))
        f_act = torch.from_numpy(b["act"].reshape(T * E, N, -1))
        f_logp = torch.from_numpy(b["logp"].reshape(T * E, N))
        fa = adv.reshape(T * E)
        f_adv = torch.from_numpy(((fa - fa.mean()) / (fa.std() + 1e-8)).astype(np.float32))
        f_tgt = torch.from_numpy(self.vnorm.norm(ret.reshape(T * E)).astype(np.float32))
        nmb = int(hp["minibatches"])
        mb = (T * E) // nmb
        st = dict(pi=[], v=[], kl=[], cf=[])
        ratio0 = None
        for _ in range(int(hp["epochs"])):
            perm = torch.randperm(T * E, generator=self.gen)
            for k in range(nmb):
                idx = perm[k * mb:(k + 1) * mb]
                o, ac, lp_old, ad = f_obs[idx], f_act[idx], f_logp[idx], f_adv[idx][:, None]
                lp = self.actor.log_prob(o, ac)
                ratio = (lp - lp_old).exp()
                if ratio0 is None:          # before any gradient step: must be exactly 1
                    ratio0 = float((ratio.detach() - 1).abs().max())
                pi_loss = -torch.min(ratio * ad, ratio.clamp(1 - clip, 1 + clip) * ad).mean()
                self.opt_a.zero_grad()
                (pi_loss - float(hp["entropy_coef"]) * self.actor.log_std.sum()).backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), float(hp["max_grad_norm"]))
                self.opt_a.step()
                v_loss = ((self.critic(o.reshape(len(idx), -1)) - f_tgt[idx]) ** 2).mean()
                self.opt_c.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), float(hp["max_grad_norm"]))
                self.opt_c.step()
                with torch.no_grad():
                    lr_ = lp - lp_old
                    st["kl"].append(float(((lr_.exp() - 1) - lr_).mean()))
                    st["cf"].append(float(((ratio - 1).abs() > clip).float().mean()))
                st["pi"].append(pi_loss.item()); st["v"].append(v_loss.item())
        return dict(pi_loss=np.mean(st["pi"]), v_loss=np.mean(st["v"]), approx_kl=np.mean(st["kl"]),
                    clipfrac=np.mean(st["cf"]), ratio0_maxdev=ratio0)

    def state(self):
        return dict(actor=self.actor.state_dict(), critic=self.critic.state_dict(), opt_a=self.opt_a.state_dict(),
                    opt_c=self.opt_c.state_dict(), vnorm=self.vnorm.state(), gen=self.gen.get_state())

    def load(self, st):
        self.actor.load_state_dict(st["actor"]); self.critic.load_state_dict(st["critic"])
        self.opt_a.load_state_dict(st["opt_a"]); self.opt_c.load_state_dict(st["opt_c"])
        self.vnorm.load(st["vnorm"]); self.gen.set_state(st["gen"])


def train(cfg: StudyConfig, run_seed: int, out_dir, overrides=None, resume=False, log=print, densities=None):
    """Formation training. densities=None trains on the study densities; anything else is a
    diagnostic run (recorded in every checkpoint as train_densities)."""
    hp = dict(cfg.mappo)
    hp.update(overrides or {})
    out = Path(out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    env = FormationTrainEnv(cfg, run_seed, int(hp["envs_per_density"]), densities=densities)   # VMAS seeds global RNGs here
    oc = cfg.obs_config()
    D, N = _obs_dim(oc), oc.n_agents
    lo, hi = env.lo.tolist(), env.hi.tolist()
    learner = MappoLearner(D, N, lo, hi, hp, run_seed)
    meta = dict(obs_dim=D, n_agents=N, hidden_actor=list(hp["hidden_actor"]), lo=lo, hi=hi,
                init_std_frac=float(hp["init_std_frac"]), env_fingerprint=cfg.env_fingerprint(),
                config=cfg.as_dict(), run_seed=run_seed, hyperparams=hp, train_densities=[g.K for g in env.groups])
    it0, env_steps, wall0 = 0, 0, 0.0
    if resume and (out / "latest.pt").exists():
        st = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        learner.load(st)
        learner.set_hp(hp)          # a resumed phase may change the recipe
        it0, env_steps, wall0 = st["iteration"], st["env_steps"], st["wall"]
        for g, idx in zip(env.groups, st["stream_idx"]):
            g.stream.idx = idx
        log(f"resumed at iteration {it0} with hyperparameters {hp}")
    # one record per training phase: run.json for the first, run_resume_itNNNN.json for each resume
    phase = "run.json" if it0 == 0 else f"run_resume_it{it0:04d}.json"
    (out / phase).write_text(json.dumps(dict(run_seed=run_seed, start_iteration=it0, env_fingerprint=meta["env_fingerprint"],
                                             obs_dim=D, hyperparams=hp, config=meta["config"]), indent=1, default=str))

    T, E = int(hp["rollout_steps"]), env.n_envs
    curve_path = out / "curve.csv"
    densities = [g.K for g in env.groups]
    cols = (["iteration", "env_steps", "wall_s", "episodes", "goal_rate", "collision_rate", "timeout_rate",
             "ep_return", "ep_len", "form_err_mean"] +
            [f"{m}_K{k}" for k in densities for m in ("goal", "coll")] +
            ["pi_loss", "v_loss", "approx_kl", "clipfrac", "std_v", "std_w", "fps"])
    if not curve_path.exists() or not resume:
        with open(curve_path, "w", newline="") as f:
            csv.writer(f).writerow(cols)
    else:   # drop rows logged after the restored checkpoint (a run stopped between checkpoints)
        with open(curve_path, newline="") as f:
            kept = [r for r in csv.reader(f)]
        kept = kept[:1] + [r for r in kept[1:] if int(r[0]) <= it0]
        with open(curve_path, "w", newline="") as f:
            csv.writer(f).writerows(kept)

    obs = env.reset()
    t_start = time.time() - wall0
    for it in range(it0, int(hp["max_iterations"])):
        t_it = time.time()
        batch, obs, episodes = learner.collect(env, obs, T)
        stats = learner.update(batch)
        env_steps += T * E
        wall = time.time() - t_start
        row = _episode_row(it + 1, env_steps, wall, episodes, densities)
        sd = learner.actor.std().detach().numpy()
        row += [stats["pi_loss"], stats["v_loss"], stats["approx_kl"], stats["clipfrac"],
                float(sd[0]), float(sd[1]), round(T * E / (time.time() - t_it))]
        with open(curve_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{x:.5g}" if isinstance(x, float) else x for x in row])
        log(f"it {it + 1} steps {env_steps} wall {wall:.0f}s eps {row[3]} goal {row[4]:.3f} coll {row[5]:.3f} "
            f"timeout {row[6]:.3f} ret {row[7]:.2f} len {row[8]:.1f} kl {stats['approx_kl']:.4f} std {sd.round(3)}")

        done_training = (it + 1 == int(hp["max_iterations"])) or wall > 60.0 * float(hp["time_limit_min"])
        if (it + 1) % int(hp["checkpoint_every"]) == 0 or done_training:
            save_actor(out / "checkpoints" / f"ckpt_it{it + 1:04d}.pt", learner.actor,
                       dict(meta, iteration=it + 1, env_steps=env_steps, wall_s=wall))
            torch.save(dict(learner.state(), iteration=it + 1, env_steps=env_steps, wall=wall,
                            stream_idx=[g.stream.idx for g in env.groups]), out / "latest.pt")
        if done_training:
            break
    return out


def _episode_row(it, env_steps, wall, episodes, densities):
    n_ep = len(episodes)
    oc_arr = np.array([e["outcome"] for e in episodes]) if n_ep else np.zeros(0)
    dens_arr = np.array([e["density"] for e in episodes]) if n_ep else np.zeros(0)
    rate = lambda o, sel=slice(None): float((oc_arr[sel] == o).mean()) if len(oc_arr[sel]) else float("nan")
    mean = lambda k: float(np.mean([e[k] for e in episodes])) if n_ep else float("nan")
    row = [it, env_steps, round(wall, 1), n_ep, rate(Outcome.GOAL), rate(Outcome.COLLISION), rate(Outcome.TIMEOUT),
           mean("ret"), mean("length"), mean("form_err_mean")]
    for k in densities:
        sel = dens_arr == k
        row += [rate(Outcome.GOAL, sel), rate(Outcome.COLLISION, sel)]
    return row
