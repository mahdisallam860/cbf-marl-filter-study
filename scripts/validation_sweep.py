#!/usr/bin/env python3
"""Locked development sweep (docs/PREFILTER_VALIDATION_PROTOCOL_LOCKED_2026-10-25.md, D4).

Nine fixed actors x K in {1, 2, 3} (selection) and {4, 5, 8} (diagnostic / stress, never used for
selection) on the EXISTING validation layouts (seed validation_base + K, 200 per density). Per
(actor, K), on identical layouts: filter off; filter on at every alpha in the grid; B-match with the
barrier rows of the same alpha. Protocol, filter, scoring, tolerances, actors and config unchanged;
a pass-through recorder adds the realised poses for the locked metrics.

One output file per (actor, K) cell under results/validation_sweep/cells/ (resumable). Paired
metrics (common-prefix formation difference vs filter off, and filter on vs B-match) are computed
inside the cell, where the per-period series are available.

Reserved blocks 4,000,000-5,999,999 are never touched: only validation seeds are used (asserted)."""
import argparse, csv, gzip, hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "validation_sweep"
ROLES = {1: "primary", 2: "primary", 3: "primary", 4: "diagnostic", 5: "stress", 8: "stress"}


def layout_hash(lay):
    h = hashlib.sha256()
    for a in (lay.init_poses, lay.obst_xy, lay.goal, lay.slots):
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def run_cell(args):
    pol_rec, K = args
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from cbf_study.bmatch import barrier_rows, rows_violated, run_bmatch
    from cbf_study.cbf_qp import barrier_point
    from cbf_study.config import StudyConfig, check_paired
    from cbf_study.evaluate import eval_layout, vmas_factory
    from cbf_study.policies import FrozenMAPPOPolicy
    from cbf_study.protocol import Outcome, run_trials
    from cbf_study.trial_metrics import (RecordingEnv, centroid_distance_series, common_prefix, formation_series,
                                         full_trajectory_mean, reached_goal_region, sampled_barrier_minima)

    cfg = StudyConfig.load()
    seed_used = cfg.validation_seed(K)
    assert not 4_000_000 <= seed_used <= 5_999_999
    pol = FrozenMAPPOPolicy(ROOT / pol_rec["ckpt"], cfg)
    assert pol.digest == pol_rec["actor_param_digest"]
    lay = eval_layout(cfg, K, "validation", 200)
    pc = cfg.protocol_config()
    fc_off = cfg.filter_config(cfg.alphas[0])
    T, (B, N) = pc.max_steps, lay.init_poses.shape[:2]
    lh = layout_hash(lay)

    def run(cond, alpha):
        fc = fc_off if alpha is None else cfg.filter_config(alpha)
        check_paired(pc, fc_off, cfg.protocol_config(), fc)
        env = RecordingEnv(vmas_factory(cfg, B, K)())
        t0 = time.time()
        if cond == "bmatch":
            res = run_bmatch(env, pol, lay, pc, fc).res
        else:
            res = run_trials(env, pol, lay, pc, fc, use_filter=(cond == "on"), log_steps=True)
        wall = time.time() - t0
        pol.assert_unchanged()
        traj = env.trajectory(T)
        Ta = res.end_step.astype(int)
        e = formation_series(traj, lay.slots)
        cd = centroid_distance_series(traj, lay.goal)
        hos, hrs = sampled_barrier_minima(traj, Ta, lay.obst_xy, fc)
        p = barrier_point(traj, fc.ell)                                          # [T+1,B,N,2]
        i, j = np.triu_indices(N, 1)
        sep = np.linalg.norm(p[:, :, i] - p[:, :, j], axis=3).min(axis=2)        # [T+1,B] min pair separation
        sep_min = np.array([sep[:Ta[b] + 1, b].min() for b in range(B)])
        # share of executed robot-periods whose clipped NOMINAL command violates an enabled robot-robot row
        # (the rows of this condition's gain; for filter off the rows of alpha = 0.5 are reported as n/a)
        rr_frac = np.full(B, np.nan)
        if alpha is not None:
            clipped = res.step_log["clipped"]
            hits = np.zeros(B)
            for k in range(int(Ta.max())):
                ex = k < Ta
                if not ex.any():
                    break
                A, bb, m = barrier_rows(fc, traj[k], lay.obst_xy)
                Kn = lay.obst_xy.shape[1]
                v = rows_violated(np.nan_to_num(clipped[k]), A[:, :, Kn:], bb[:, :, Kn:], m[:, :, Kn:], fc.solver_tol)
                hits += np.where(ex, v.sum(axis=1), 0)
            rr_frac = np.where(Ta > 0, hits / np.maximum(Ta * N, 1), np.nan)
        return dict(cond=cond, alpha=alpha, res=res, e=e, cd=cd, T=Ta, wall=wall, hos=hos, hrs=hrs, sep_min=sep_min,
                    sep0=sep[0], rr_frac=rr_frac, ftm=full_trajectory_mean(e, Ta),
                    rgr=reached_goal_region(cd, Ta, res.outcome, 0.30))

    runs = [run("off", None)]
    for a in cfg.alphas:
        runs += [run("on", a), run("bmatch", a)]
    off = runs[0]
    on_by = {r["alpha"]: r for r in runs if r["cond"] == "on"}
    rows = []
    for r in runs:
        res = r["res"]
        d_off, L_off, why_off = common_prefix(off["e"], off["T"], r["e"], r["T"])
        if r["cond"] == "bmatch":                   # filter on vs B-match at the same gain: d = e_on - e_bmatch
            o = on_by[r["alpha"]]
            d_ob, L_ob, why_ob = common_prefix(r["e"], r["T"], o["e"], o["T"])
        else:
            d_ob, L_ob, why_ob = np.full(B, np.nan), np.zeros(B, int), np.full(B, "n/a", dtype=object)
        for b in range(B):
            oc = Outcome(res.outcome[b]).name
            rows.append(dict(
                policy_id=pol_rec["policy_id"], seed=pol_rec["seed"], phase=pol_rec["phase"], density=K, role=ROLES[K],
                condition=r["cond"], alpha="" if r["alpha"] is None else r["alpha"], trial=b, layout_seed=seed_used,
                layout_hash=lh, outcome=oc, collision_kind={0: "none", 1: "obstacle", 2: "robot"}[int(res.collision_kind[b])],
                goal=int(oc == "GOAL"), fail=int(oc in ("COLLISION", "INFEASIBLE")), executed_periods=int(r["T"][b]),
                stalled=int(res.stalled[b]), goal_dist_final=float(res.goal_dist_final[b]),
                form_err_final=float(res.form_err_final[b]), full_traj_form_mean=float(r["ftm"][b]),
                reached_goal_region=int(r["rgr"][b]),
                n_interventions=int(res.n_interventions[b]) if r["cond"] == "on" else 0,
                n_brake_activations=int(res.n_interventions[b]) if r["cond"] == "bmatch" else 0,
                n_infeasible_robot_steps=int(res.n_infeasible[b]),
                h_obst_min_prestep=float(res.h_obst_min[b]), h_rr_min_prestep=float(res.h_rr_min[b]),
                h_obst_min_sampled=float(r["hos"][b]), h_rr_min_sampled=float(r["hrs"][b]),
                bp_sep_min=float(r["sep_min"][b]), bp_sep_initial=float(r["sep0"][b]), rr_nominal_violation_frac=float(r["rr_frac"][b]),
                d_vs_off=float(d_off[b]), L_vs_off=int(L_off[b]), excl_vs_off=why_off[b],
                d_on_minus_bmatch=float(d_ob[b]), L_on_bmatch=int(L_ob[b]), excl_on_bmatch=why_ob[b],
                track_err_max=float(res.track_err_max[b])))
    meta = dict(policy_id=pol_rec["policy_id"], density=K, layout_seed=seed_used, layout_hash=lh, actor_digest=pol.digest,
                walls={f"{r['cond']}_{r['alpha']}": round(r["wall"], 2) for r in runs})
    return pol_rec["policy_id"], K, rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from cbf_study.config import StudyConfig
    cfg = StudyConfig.load()
    assert cfg.env_fingerprint() == "c60352aee3f11ed8"
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())
    cells = OUT / "cells"
    cells.mkdir(parents=True, exist_ok=True)
    order = [1, 2, 3, 4, 5, 8]
    tasks = [(p, K) for K in order for p in sel["policies"]
             if not (cells / f"{p['policy_id']}_K{K}.csv.gz").exists()]
    print(f"{len(tasks)} cells to run", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_cell, t) for t in tasks]
        for f in as_completed(futs):
            pid, K, rows, meta = f.result()
            tmp = cells / f"{pid}_K{K}.csv.gz.tmp"
            with gzip.open(tmp, "wt", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows({k: (f"{v:.7g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
            (cells / f"{pid}_K{K}.meta.json").write_text(json.dumps(meta, indent=1))
            tmp.rename(cells / f"{pid}_K{K}.csv.gz")
            print(f"{pid} K{K} done ({sum(meta['walls'].values()):.0f} s run time) elapsed {time.time() - t0:.0f} s", flush=True)
    (OUT / "sweep_meta.json").write_text(json.dumps(dict(
        protocol="docs/PREFILTER_VALIDATION_PROTOCOL_LOCKED_2026-10-25.md", protocol_commit="9baf6d7",
        env_fingerprint=cfg.env_fingerprint(), alphas=cfg.alphas, densities=order, roles=ROLES, trials_per_cell=200,
        split="validation", workers=a.workers, wall_s_this_invocation=round(time.time() - t0, 1)), indent=1))


if __name__ == "__main__":
    main()
