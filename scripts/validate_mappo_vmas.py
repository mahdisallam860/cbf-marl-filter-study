#!/usr/bin/env python3
"""Validate the MAPPO implementation on a stock VMAS scenario (navigation, 3 agents) before any
retraining on the formation task. Same MappoLearner and recipe as the formation runs.

Reports whether the deterministic policy learns beyond its initial policy (and beyond uniform
random actions) on a FIXED set of evaluation episodes (separate seed), and logs the per-update
checks: PPO ratio before the first gradient step, KL, clip fraction, applied-action bounds.
Writes results/mappo_validation/{curve.csv, eval.csv, run.json, VALIDATION.md}."""
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch


def evaluate(policy_fn, n_envs, max_steps, seed, n_agents):
    """First episode of each of n_envs fixed evaluation envs; policy_fn(obs) -> raw actions."""
    from cbf_study.stock_env import VmasStockEnv
    env = VmasStockEnv(n_envs=n_envs, n_agents=n_agents, max_steps=max_steps, seed=seed)
    obs = env.reset()
    ret, length, succ, done = np.zeros(n_envs), np.zeros(n_envs, int), np.zeros(n_envs, bool), np.zeros(n_envs, bool)
    for _ in range(max_steps):
        so = env.step(policy_fn(obs))
        live = ~done
        ret[live] += so.reward[live]; length[live] += 1
        fin = live & (so.terminated | so.truncated)
        succ[fin] = so.terminated[fin]
        done |= fin
        obs = so.obs
        if done.all():
            break
    return dict(ret_mean=float(ret.mean()), ret_sd=float(ret.std(ddof=1)), success=float(succ.mean()),
                len_mean=float(length.mean()), n=n_envs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--time-limit-min", type=float, default=12.0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-envs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-seed", type=int, default=777)
    ap.add_argument("--out", default="results/mappo_validation")
    a = ap.parse_args()
    from cbf_study.config import StudyConfig
    from cbf_study.mappo import MappoLearner
    from cbf_study.stock_env import VmasStockEnv
    torch.set_num_threads(1)
    cfg = StudyConfig.load()
    hp = dict(cfg.mappo, epochs=10, minibatches=16)          # the recipe of training phase 2
    N = 3
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    env = VmasStockEnv(n_envs=a.envs, n_agents=N, max_steps=a.max_steps, seed=a.seed)
    obs = env.reset()
    D = obs.shape[2]
    lrn = MappoLearner(D, N, env.lo, env.hi, hp, seed=a.seed)
    det = lambda o: lrn.actor.mean(torch.from_numpy(o)).detach().numpy()
    rng = np.random.default_rng(a.seed)
    evals = [dict(iteration=0, env_steps=0, policy="uniform_random",
                  **evaluate(lambda o: rng.uniform(env.lo, env.hi, o.shape[:2] + (len(env.lo),)), a.eval_envs, a.max_steps, a.eval_seed, N)),
             dict(iteration=0, env_steps=0, policy="initial_deterministic", **evaluate(det, a.eval_envs, a.max_steps, a.eval_seed, N))]
    print(evals, flush=True)
    cols = ["iteration", "env_steps", "wall_s", "episodes", "train_success", "train_ret", "train_len", "pi_loss", "v_loss",
            "approx_kl", "clipfrac", "ratio0_maxdev", "applied_min", "applied_max", "std0", "std1"]
    rows, t0, steps = [], time.time(), 0
    T = int(hp["rollout_steps"])
    for it in range(1, a.iterations + 1):
        applied = []
        orig = env.step
        env.step = lambda raw: (lambda so: (applied.append(env.last_applied), so)[1])(orig(raw))
        b, obs, eps = lrn.collect(env, obs, T)
        env.step = orig
        st = lrn.update(b)
        steps += T * a.envs
        app = np.concatenate(applied)
        sd = lrn.actor.std().detach().numpy()
        rows.append(dict(iteration=it, env_steps=steps, wall_s=round(time.time() - t0, 1), episodes=len(eps),
                         train_success=float(np.mean([e["success"] for e in eps])) if eps else float("nan"),
                         train_ret=float(np.mean([e["ret"] for e in eps])) if eps else float("nan"),
                         train_len=float(np.mean([e["length"] for e in eps])) if eps else float("nan"),
                         pi_loss=st["pi_loss"], v_loss=st["v_loss"], approx_kl=st["approx_kl"], clipfrac=st["clipfrac"],
                         ratio0_maxdev=st["ratio0_maxdev"], applied_min=float(app.min()), applied_max=float(app.max()),
                         std0=float(sd[0]), std1=float(sd[1])))
        r = rows[-1]
        print(f"it {it} steps {steps} succ {r['train_success']:.3f} ret {r['train_ret']:.3f} kl {r['approx_kl']:.4f} "
              f"ratio0 {r['ratio0_maxdev']:.1e}", flush=True)
        last = it == a.iterations or time.time() - t0 > 60 * a.time_limit_min
        if it % a.eval_every == 0 or last:
            evals.append(dict(iteration=it, env_steps=steps, policy="deterministic", **evaluate(det, a.eval_envs, a.max_steps, a.eval_seed, N)))
            print(evals[-1], flush=True)
        if last:
            break
    for name, data in (("curve.csv", rows), ("eval.csv", evals)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    lo, hi = env.lo.tolist(), env.hi.tolist()
    summary = dict(scenario="navigation", n_agents=N, envs=a.envs, max_steps=a.max_steps, hyperparams=hp, seed=a.seed,
                   eval_seed=a.eval_seed, action_box=[lo, hi], iterations=len(rows), env_steps=steps,
                   ratio0_maxdev_max=max(r["ratio0_maxdev"] for r in rows),
                   applied_within_box=bool(min(r["applied_min"] for r in rows) >= min(lo) - 1e-6 and
                                           max(r["applied_max"] for r in rows) <= max(hi) + 1e-6),
                   initial=evals[1], random=evals[0], final=evals[-1],
                   best=max(evals[2:], key=lambda e: e["ret_mean"]) if len(evals) > 2 else None)
    (out / "run.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
