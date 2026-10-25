"""
Compact MAPPO (PyTorch) for the formation-navigation task. Training runs WITHOUT the filter.

* Actor: one network shared by all robots (agent id is part of the observation). Gaussian policy
  whose mean is center + half * tanh(net(obs)), so the deterministic action (the mean) always lies
  in the velocity box; a state-independent log-std. Sampled actions are clipped to the box by the
  environment, exactly as the protocol clips commands.
* Critic: centralised, V(concat of all robots' observations) for the shared team reward.
* PPO: GAE(gamma, lambda), clipped surrogate, advantage normalisation, running value
  normalisation, separate Adam optimisers, gradient-norm clipping. Timeouts are truncations:
  r += gamma * V(final state) before the episode is reset.

Checkpoints (actor only, for evaluation) are written every `checkpoint_every` iterations and at
the end; latest.pt holds the full training state for resuming.
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
        self.body = mlp(obs_dim, hidden, 2, 0.01)
        lo = torch.as_tensor(lo, dtype=torch.float32)
        hi = torch.as_tensor(hi, dtype=torch.float32)
        self.register_buffer("a_center", (hi + lo) / 2)
        self.register_buffer("a_half", (hi - lo) / 2)
        self.log_std = nn.Parameter(torch.full((2,), float(np.log(init_std_frac))))

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


def train(cfg: StudyConfig, run_seed: int, out_dir, overrides=None, resume=False, log=print):
    hp = dict(cfg.mappo)
    hp.update(overrides or {})
    out = Path(out_dir)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    env = FormationTrainEnv(cfg, run_seed, int(hp["envs_per_density"]))   # VMAS seeds global RNGs here
    torch.manual_seed(run_seed)
    gen = torch.Generator().manual_seed(run_seed)
    oc = cfg.obs_config()
    D, N = _obs_dim(oc), oc.n_agents
    lo, hi = env.lo.tolist(), env.hi.tolist()
    meta = dict(obs_dim=D, n_agents=N, hidden_actor=list(hp["hidden_actor"]), lo=lo, hi=hi,
                init_std_frac=float(hp["init_std_frac"]), env_fingerprint=cfg.env_fingerprint(),
                config=cfg.as_dict(), run_seed=run_seed, hyperparams=hp)
    actor = build_actor(meta)
    critic = Critic(N * D, list(hp["hidden_critic"]))
    opt_a = torch.optim.Adam(actor.parameters(), lr=float(hp["lr_actor"]), eps=1e-5)
    opt_c = torch.optim.Adam(critic.parameters(), lr=float(hp["lr_critic"]), eps=1e-5)
    vnorm = ValueNorm()
    it0, env_steps, wall0 = 0, 0, 0.0
    if resume and (out / "latest.pt").exists():
        st = torch.load(out / "latest.pt", map_location="cpu", weights_only=False)
        actor.load_state_dict(st["actor"]); critic.load_state_dict(st["critic"])
        opt_a.load_state_dict(st["opt_a"]); opt_c.load_state_dict(st["opt_c"])
        vnorm.load(st["vnorm"]); gen.set_state(st["gen"])
        it0, env_steps, wall0 = st["iteration"], st["env_steps"], st["wall"]
        for g, idx in zip(env.groups, st["stream_idx"]):
            g.stream.idx = idx
        log(f"resumed at iteration {it0}")
    (out / "run.json").write_text(json.dumps(dict(run_seed=run_seed, env_fingerprint=meta["env_fingerprint"],
                                                  obs_dim=D, hyperparams=hp, config=meta["config"]), indent=1, default=str))

    T, E = int(hp["rollout_steps"]), env.n_envs
    gamma, lam, clip = float(hp["gamma"]), float(hp["gae_lambda"]), float(hp["clip"])
    curve_path = out / "curve.csv"
    densities = [g.K for g in env.groups]
    cols = (["iteration", "env_steps", "wall_s", "episodes", "goal_rate", "collision_rate", "timeout_rate",
             "ep_return", "ep_len", "form_err_mean"] +
            [f"{m}_K{k}" for k in densities for m in ("goal", "coll")] +
            ["pi_loss", "v_loss", "approx_kl", "clipfrac", "std_v", "std_w", "fps"])
    if not curve_path.exists() or not resume:
        with open(curve_path, "w", newline="") as f:
            csv.writer(f).writerow(cols)

    def joint(o):
        return o.reshape(o.shape[0], -1)

    obs = env.reset()
    t_start = time.time() - wall0
    it = it0
    for it in range(it0, int(hp["max_iterations"])):
        t_it = time.time()
        b_obs = np.zeros((T, E, N, D), np.float32)
        b_act = np.zeros((T, E, N, 2), np.float32)
        b_logp = np.zeros((T, E, N), np.float32)
        b_rew = np.zeros((T, E))
        b_done = np.zeros((T, E))
        b_val = np.zeros((T, E))
        episodes = []
        for t in range(T):
            ot = torch.from_numpy(obs)
            with torch.no_grad():
                mu = actor.mean(ot)
                a = mu + actor.std() * torch.randn(mu.shape, generator=gen)
                logp = torch.distributions.Normal(mu, actor.std()).log_prob(a).sum(-1)
                v = vnorm.denorm(critic(torch.from_numpy(joint(obs))).double().numpy())
            so = env.step(a.numpy())
            r = so.reward.copy()
            if so.truncated.any():
                with torch.no_grad():
                    fv = vnorm.denorm(critic(torch.from_numpy(joint(so.final_obs))).double().numpy())
                r = np.where(so.truncated, r + gamma * fv, r)
            b_obs[t], b_act[t], b_logp[t] = obs, a.numpy(), logp.numpy()
            b_rew[t], b_done[t], b_val[t] = r, (so.terminated | so.truncated), v
            episodes += so.episodes
            obs = so.obs
        with torch.no_grad():
            last_v = vnorm.denorm(critic(torch.from_numpy(joint(obs))).double().numpy())
        adv = _gae(b_rew, b_val, b_done, last_v, gamma, lam)
        ret = adv + b_val
        vnorm.update(ret)
        env_steps += T * E

        f_obs = torch.from_numpy(b_obs.reshape(T * E, N, D))
        f_act = torch.from_numpy(b_act.reshape(T * E, N, 2))
        f_logp = torch.from_numpy(b_logp.reshape(T * E, N))
        f_adv = adv.reshape(T * E)
        f_adv = torch.from_numpy(((f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)).astype(np.float32))
        f_tgt = torch.from_numpy(vnorm.norm(ret.reshape(T * E)).astype(np.float32))
        mb = (T * E) // int(hp["minibatches"])
        stats = dict(pi=[], v=[], kl=[], cf=[])
        for _ in range(int(hp["epochs"])):
            perm = torch.randperm(T * E, generator=gen)
            for k in range(int(hp["minibatches"])):
                idx = perm[k * mb:(k + 1) * mb]
                o, ac, lp_old, ad = f_obs[idx], f_act[idx], f_logp[idx], f_adv[idx]
                lp = actor.log_prob(o, ac)
                ratio = (lp - lp_old).exp()
                ad = ad[:, None]
                pi_loss = -torch.min(ratio * ad, ratio.clamp(1 - clip, 1 + clip) * ad).mean()
                ent = actor.log_std.sum()
                opt_a.zero_grad()
                (pi_loss - float(hp["entropy_coef"]) * ent).backward()
                nn.utils.clip_grad_norm_(actor.parameters(), float(hp["max_grad_norm"]))
                opt_a.step()
                v_loss = ((critic(o.reshape(len(idx), -1)) - f_tgt[idx]) ** 2).mean()
                opt_c.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), float(hp["max_grad_norm"]))
                opt_c.step()
                with torch.no_grad():
                    lr_ = lp - lp_old
                    stats["kl"].append(float(((lr_.exp() - 1) - lr_).mean()))
                    stats["cf"].append(float(((ratio - 1).abs() > clip).float().mean()))
                stats["pi"].append(pi_loss.item()); stats["v"].append(v_loss.item())

        # ---- log ---------------------------------------------------------------------------------
        wall = time.time() - t_start
        n_ep = len(episodes)
        oc_arr = np.array([e["outcome"] for e in episodes]) if n_ep else np.zeros(0)
        rate = lambda o, sel=slice(None): float((oc_arr[sel] == o).mean()) if len(oc_arr[sel]) else float("nan")
        dens_arr = np.array([e["density"] for e in episodes]) if n_ep else np.zeros(0)
        row = [it + 1, env_steps, round(wall, 1), n_ep, rate(Outcome.GOAL), rate(Outcome.COLLISION), rate(Outcome.TIMEOUT),
               float(np.mean([e["ret"] for e in episodes])) if n_ep else float("nan"),
               float(np.mean([e["length"] for e in episodes])) if n_ep else float("nan"),
               float(np.mean([e["form_err_mean"] for e in episodes])) if n_ep else float("nan")]
        for k in densities:
            sel = dens_arr == k
            row += [rate(Outcome.GOAL, sel), rate(Outcome.COLLISION, sel)]
        sd = actor.std().detach().numpy()
        row += [np.mean(stats["pi"]), np.mean(stats["v"]), np.mean(stats["kl"]), np.mean(stats["cf"]),
                float(sd[0]), float(sd[1]), round(T * E / (time.time() - t_it))]
        with open(curve_path, "a", newline="") as f:
            csv.writer(f).writerow([f"{x:.5g}" if isinstance(x, float) else x for x in row])
        log(f"it {it + 1} steps {env_steps} wall {wall:.0f}s eps {n_ep} goal {row[4]:.3f} coll {row[5]:.3f} "
            f"timeout {row[6]:.3f} ret {row[7]:.2f} len {row[8]:.1f} kl {row[-5]:.4f} std {sd.round(3)}")

        done_training = (it + 1 == int(hp["max_iterations"])) or wall > 60.0 * float(hp["time_limit_min"])
        if (it + 1) % int(hp["checkpoint_every"]) == 0 or done_training:
            save_actor(out / "checkpoints" / f"ckpt_it{it + 1:04d}.pt", actor,
                       dict(meta, iteration=it + 1, env_steps=env_steps, wall_s=wall))
            torch.save(dict(actor=actor.state_dict(), critic=critic.state_dict(), opt_a=opt_a.state_dict(),
                            opt_c=opt_c.state_dict(), vnorm=vnorm.state(), gen=gen.get_state(),
                            iteration=it + 1, env_steps=env_steps, wall=wall,
                            stream_idx=[g.stream.idx for g in env.groups]), out / "latest.pt")
        if done_training:
            break
    return out
