#!/usr/bin/env python3
"""Curriculum MAPPO training (filter OFF), config v2: stages from [curriculum] in config/study.toml
(K = 0 -> 1 -> 2 -> 3 -> study densities), env-step budgets fixed in the config before any run.

* A stage ends when its budget is spent, or earlier when the held-out team GOAL rate at every stage
  density reaches advance_goal_rate (held-out = validation layouts, deterministic frozen policy,
  filter off, the gate's evaluation path).
* STOP rule: when the K=2 stage ends with a held-out K=2 GOAL rate below k2_stop_goal_rate, training
  stops and diagnostics are written (no further stages).
* Training layouts: each density keeps one seeded stream across stages, so no training layout is
  repeated and none coincides with validation or pilot layouts.

Writes results/curriculum/seed_<s>/{curve.csv, heldout.csv, stages.json, train.log,
stage_checkpoints/, checkpoints/ (final stage only: the gate reads these)}."""
import argparse, csv, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch


def heldout(cfg, ckpt, densities, n):
    from cbf_study.evaluate import eval_layout, run_condition
    from cbf_study.policies import FrozenMAPPOPolicy
    from cbf_study.protocol import Outcome
    pol = FrozenMAPPOPolicy(ckpt, cfg)
    out = {}
    for K in densities:
        lay = eval_layout(cfg, K, "validation", n)
        r = run_condition(cfg, pol, lay, K)
        o = r.outcome
        to = o == Outcome.TIMEOUT
        out[K] = dict(goal=float((o == Outcome.GOAL).mean()), collision=float((o == Outcome.COLLISION).mean()),
                      timeout=float(to.mean()), coll_obst=int(((o == Outcome.COLLISION) & (r.collision_kind == 1)).sum()),
                      coll_rr=int(((o == Outcome.COLLISION) & (r.collision_kind == 2)).sum()), stalled=int(r.stalled.sum()),
                      timeout_goal_dist_median=float(np.median(r.goal_dist_final[to])) if to.any() else float("nan"),
                      form_err_mean=float(r.form_err_mean.mean()), n=n, layout_seed=cfg.validation_seed(K))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--smoke", action="store_true", help="TEST ONLY: tiny budgets, envs and evaluations")
    a = ap.parse_args()
    from cbf_study.config import StudyConfig
    from cbf_study.mappo import MappoLearner, _episode_row, save_actor
    from cbf_study.observations import obs_dim
    from cbf_study.train_env import FormationTrainEnv
    torch.set_num_threads(1)
    cfg = StudyConfig.load(a.config) if a.config else StudyConfig.load()
    cur, hp = cfg.curriculum(), dict(cfg.mappo)
    if a.smoke:
        cur = dict(cur, budget=[2 * 8 * 12] * len(cur["stages"]), envs_per_iter=12, eval_trials=4, eval_every=1)
        hp.update(rollout_steps=8, epochs=1, minibatches=2)
    out = Path(a.out or f"results/curriculum/seed_{a.seed}")
    for d in ("checkpoints", "stage_checkpoints"):
        (out / d).mkdir(parents=True, exist_ok=True)
    logf = open(out / "train.log", "a")
    def log(msg):
        print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
    oc = cfg.obs_config()
    D, N = obs_dim(oc), oc.n_agents
    fc = cfg.layout_filter_config()
    lo, hi = [fc.v_min, -fc.w_max], [fc.v_max, fc.w_max]
    learner = MappoLearner(D, N, lo, hi, hp, a.seed)
    meta = dict(obs_dim=D, n_agents=N, hidden_actor=list(hp["hidden_actor"]), lo=lo, hi=hi,
                init_std_frac=float(hp["init_std_frac"]), env_fingerprint=cfg.env_fingerprint(), config=cfg.as_dict(),
                run_seed=a.seed, hyperparams=hp, curriculum=cur)
    (out / "run.json").write_text(json.dumps(dict(seed=a.seed, config=cfg.path, config_version=cfg.version,
                                                  env_fingerprint=cfg.env_fingerprint(), curriculum=cur, hyperparams=hp), indent=1))
    log(f"seed {a.seed} config v{cfg.version} fingerprint {cfg.env_fingerprint()} curriculum {cur}")
    T = int(hp["rollout_steps"])
    stream_idx = {}
    total, it_global, t0 = 0, 0, time.time()
    stages, heldout_rows, curve = [], [], []
    stopped = None
    for si, (dens, budget) in enumerate(zip(cur["stages"], cur["budget"])):
        final_stage = si == len(cur["stages"]) - 1
        E_d = cur["envs_per_iter"] // len(dens)
        env = FormationTrainEnv(cfg, a.seed, E_d, densities=dens, vmas_seed=a.seed * 100 + si)
        for g in env.groups:
            g.stream.idx = stream_idx.get(g.K, 0)
        obs = env.reset()
        steps, it, last_eval = 0, 0, None
        log(f"stage {si} densities {dens} budget {budget} env steps, {E_d} envs per density, cumulative {total}")
        while steps < budget:
            b, obs, eps = learner.collect(env, obs, T)
            st = learner.update(b)
            steps += T * env.n_envs; total += T * env.n_envs; it += 1; it_global += 1
            row = _episode_row(it_global, total, time.time() - t0, eps, list(dens))
            curve.append(dict(stage=si, densities="/".join(map(str, dens)), stage_env_steps=steps,
                              **dict(zip(["iteration", "env_steps", "wall_s", "episodes", "goal_rate", "collision_rate",
                                          "timeout_rate", "ep_return", "ep_len", "form_err_mean"], row[:10])),
                              approx_kl=st["approx_kl"], clipfrac=st["clipfrac"], ratio0_maxdev=st["ratio0_maxdev"]))
            log(f"stage {si} it {it} steps {steps}/{budget} total {total} goal {row[4]:.3f} coll {row[5]:.3f} "
                f"timeout {row[6]:.3f} kl {st['approx_kl']:.4f}")
            end = steps >= budget
            if it % cur["eval_every"] == 0 or end:
                name = f"stage{si}_K{'-'.join(map(str, dens))}_it{it_global:04d}.pt"
                ck = out / "stage_checkpoints" / name
                save_actor(ck, learner.actor, dict(meta, iteration=it_global, env_steps=total, stage=si, train_densities=list(dens)))
                if final_stage:
                    save_actor(out / "checkpoints" / f"ckpt_it{it_global:04d}.pt", learner.actor,
                               dict(meta, iteration=it_global, env_steps=total, stage=si, train_densities=list(dens)))
                ev = heldout(cfg, ck, dens, cur["eval_trials"])
                last_eval = ev
                for K, e in ev.items():
                    heldout_rows.append(dict(stage=si, densities="/".join(map(str, dens)), iteration=it_global,
                                             stage_env_steps=steps, env_steps=total, density=K, ckpt=str(ck), **e))
                log(f"  held-out {', '.join(f'K{K}: goal {e['goal']:.3f} coll {e['collision']:.3f} timeout {e['timeout']:.3f}' for K, e in ev.items())}")
                if all(e["goal"] >= cur["advance_goal_rate"] for e in ev.values()) and not final_stage:
                    log(f"  advance: held-out GOAL >= {cur['advance_goal_rate']} at every stage density")
                    break
        for g in env.groups:
            stream_idx[g.K] = g.stream.idx
        stages.append(dict(stage=si, densities=list(dens), budget=budget, env_steps_used=steps, cumulative_env_steps=total,
                           iterations=it, advanced_early=steps < budget, heldout_end=last_eval, wall_s=round(time.time() - t0, 1)))
        if dens == (2,) and last_eval[2]["goal"] < cur["k2_stop_goal_rate"]:
            stopped = (f"STOP: held-out K=2 team GOAL {last_eval[2]['goal']:.3f} < {cur['k2_stop_goal_rate']} "
                       f"after the K=2 budget ({steps} env steps, cumulative {total})")
            log(stopped)
            break
    for name, data in (("curve.csv", curve), ("heldout.csv", heldout_rows)):
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    torch.save(dict(learner.state(), env_steps=total, iteration=it_global, stream_idx=stream_idx), out / "latest.pt")
    (out / "stages.json").write_text(json.dumps(dict(stages=stages, stopped=stopped, total_env_steps=total,
                                                     wall_s=round(time.time() - t0, 1)), indent=1, default=str))
    log(json.dumps(dict(stages=[(s["stage"], s["densities"], s["env_steps_used"], s["cumulative_env_steps"]) for s in stages],
                        stopped=stopped)))


if __name__ == "__main__":
    main()
