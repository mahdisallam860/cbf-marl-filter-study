#!/usr/bin/env python3
"""Phase 1, items 2 and 3: filter-OFF evaluation of the nine fixed policies
(results/phase1/checkpoint_selection.json) on the existing VALIDATION layouts
(seeds validation_base + K, 200 trials per density):

    K = 1, 2, 3   candidate study densities
    K = 5, 8      baseline stress results
    K = 4         intermediate diagnostic (validation seed validation_base + 4, first use; its layouts
                  were audited for solvability first: results/phase1/solvability/)

Each trial goes through the unchanged protocol path (protocol.run_trials, filter off, the same
ProtocolConfig / FilterConfig and check_paired guard as evaluate.run_condition). The only addition is
a recording wrapper around VmasEnv that stores the poses after every control period; it does not
change commands, scoring or outcomes. The recorded trajectories give per-trial diagnostics for the
near-goal timeout analysis and the exploratory tolerance sensitivity (looser tolerances only: a trial
that meets a looser criterion at period t has the same trajectory up to t, so its outcome under that
criterion is exact; tighter tolerances cannot be evaluated from runs that stopped at 0.10 m).

No filter-on or brake trial is run. The study tolerances (0.10 m) are not changed.

Writes results/phase1/eval/{trials.csv, runs.csv, meta.json}."""
import argparse, hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DENSITIES = {1: "primary", 2: "primary", 3: "primary", 4: "diagnostic_K4", 5: "stress", 8: "stress"}
ALT_GOAL_TOL = (0.10, 0.125, 0.15, 0.20, 0.25)      # exploratory only
ALT_FORM_TOL = (0.10, 0.15, 0.20)                   # exploratory only
NEAR_GOAL = 0.30        # TIMEOUT with final centroid-to-goal distance <= slot radius: "near-goal timeout"
WINDOW = 50             # last control periods examined (= protocol stall_window)


class RecordingEnv:
    """Pass-through wrapper: records poses after reset and after every step."""

    def __init__(self, env):
        self.env, self.traj = env, []

    def reset(self, init_poses, obst_xy):
        self.env.reset(init_poses, obst_xy)
        self.traj = [self.env.poses()]

    def poses(self):
        return self.env.poses()

    def step(self, u):
        self.env.step(u)
        self.traj.append(self.env.poses())


def layout_hash(lay):
    h = hashlib.sha256()
    for a in (lay.init_poses, lay.obst_xy, lay.goal, lay.slots):
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def run_task(args):
    pol_rec, K = args
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from cbf_study.config import StudyConfig, check_paired
    from cbf_study.evaluate import eval_layout, vmas_factory
    from cbf_study.policies import FrozenMAPPOPolicy
    from cbf_study.protocol import Outcome, formation_error, run_trials
    from cbf_study.train_env import proximity_penalty

    cfg = StudyConfig.load()
    pol = FrozenMAPPOPolicy(ROOT / pol_rec["ckpt"], cfg)
    assert pol.digest == pol_rec["actor_param_digest"], "checkpoint differs from the fixed selection"
    lay = eval_layout(cfg, K, "validation", 200)
    pc = cfg.protocol_config()
    fc_off = cfg.filter_config(cfg.alphas[0])
    check_paired(pc, fc_off, cfg.protocol_config(), fc_off)          # as evaluate.run_condition, filter off
    env = RecordingEnv(vmas_factory(cfg, lay.init_poses.shape[0], K)())
    t0 = time.time()
    res = run_trials(env, pol, lay, pc, fc_off, use_filter=False, log_steps=True)
    wall = time.time() - t0
    pol.assert_unchanged()

    P = np.stack(env.traj)                                  # [S+1, B, N, 3]
    S = P.shape[0] - 1
    B, N = lay.init_poses.shape[:2]
    cen = P[..., :2].mean(axis=2)                           # [S+1, B, 2]
    cdist = np.linalg.norm(cen - lay.goal[None], axis=2)    # [S+1, B]
    ferr = np.stack([formation_error(P[t], lay.slots) for t in range(S + 1)])   # [S+1, B]
    app = res.step_log["applied"]                           # [T, B, N, 2], NaN when inactive
    rc = cfg.reward_config()
    goal_obst = np.linalg.norm(lay.obst_xy - lay.goal[:, None], axis=2).min(axis=1) if K else np.full(B, np.inf)
    rows = []
    for b in range(B):
        e = int(res.end_step[b])
        oc = Outcome(res.outcome[b]).name
        cd, fe = cdist[:e + 1, b], ferr[:e + 1, b]
        assert abs(cd[-1] - res.goal_dist_final[b]) < 1e-9 and abs(fe[-1] - res.form_err_final[b]) < 1e-9
        score = np.maximum(cd / pc.goal_tol, fe / pc.formation_tol)
        tstar = int(score.argmin())
        r = dict(policy_id=pol_rec["policy_id"], seed=pol_rec["seed"], phase=pol_rec["phase"], iteration=pol_rec["iteration"],
                 density=K, role=DENSITIES[K], layout_seed=cfg.validation_seed(K), trial=b, outcome=oc,
                 collision_kind={0: "none", 1: "obstacle", 2: "robot"}[int(res.collision_kind[b])], end_step=e,
                 stalled=int(res.stalled[b]), form_err_mean=float(res.form_err_mean[b]),
                 form_err_final=float(res.form_err_final[b]), goal_dist_final=float(res.goal_dist_final[b]),
                 goal_dist_min=float(cd.min()), form_err_min=float(fe.min()), goal_score_min=float(score[tstar]),
                 goal_score_argmin_step=tstar, goal_dist_at_score_min=float(cd[tstar]), form_err_at_score_min=float(fe[tstar]),
                 h_obst_min=float(res.h_obst_min[b]), h_rr_min=float(res.h_rr_min[b]),
                 nearest_obstacle_to_goal=float(goal_obst[b]))
        # final-state decomposition of each robot's error to its absolute target goal + slot_i, body frame
        th = P[e, b, :, 2]
        err = lay.goal[b] + lay.slots - P[e, b, :, :2]
        lon = np.abs(np.cos(th) * err[:, 0] + np.sin(th) * err[:, 1])
        lat = np.abs(-np.sin(th) * err[:, 0] + np.cos(th) * err[:, 1])
        gd = lay.goal[b] - cen[e, b]
        hd = np.degrees(np.arctan2(np.sin(th).mean(), np.cos(th).mean()))
        # first period from which every robot's applied command stays below 0.01 (m/s, rad/s) until the end
        moving = np.flatnonzero(np.nanmax(np.abs(app[:e, b]), axis=(1, 2)) >= 0.01) if e else np.array([], int)
        onset = (int(moving[-1]) + 1 if len(moving) else 0) if (e and (not len(moving) or moving[-1] < e - 1)) else -1
        r.update(final_goal_dx=float(gd[0]), final_goal_dy=float(gd[1]), final_heading_mean_deg=float(hd),
                 stop_onset=onset, stationary_periods=(e - onset) if onset >= 0 else 0)
        r.update(final_target_err_max=float(np.linalg.norm(err, axis=1).max()), final_lateral_mean=float(lat.mean()),
                 final_longitudinal_mean=float(lon.mean()),
                 final_proximity_penalty=float(proximity_penalty(P[e, b][None], lay.obst_xy[b][None], pc, rc.proximity_margin)[0]),
                 final_fail=("met" if (cd[-1] <= pc.goal_tol and fe[-1] <= pc.formation_tol) else
                             "centroid" if fe[-1] <= pc.formation_tol else
                             "formation" if cd[-1] <= pc.goal_tol else "both"))
        # last WINDOW periods before the end
        lo = max(0, e - WINDOW)
        if e > lo:
            a = app[lo:e, b]                                   # [w, N, 2]
            v, w = a[..., 0], a[..., 1]
            sgn = np.sign(np.where(np.abs(v) < 1e-3, 0.0, v))
            flips = sum(int(((sgn[1:, i] * sgn[:-1, i]) < 0).sum()) for i in range(N))
            raw = res.step_log["raw"][lo:e, b]
            r.update(win_raw_abs_v_max=float(np.abs(raw[..., 0]).max()), win_raw_abs_w_max=float(np.abs(raw[..., 1]).max()))
            r.update(win_progress=float(cdist[lo, b] - cdist[e, b]),
                     win_centroid_path=float(np.linalg.norm(np.diff(cen[lo:e + 1, b], axis=0), axis=1).sum()),
                     win_abs_v=float(np.abs(v).mean()), win_abs_w=float(np.abs(w).mean()),
                     win_frac_v_small=float((np.abs(v) < 0.02).mean()), win_v_sign_flips=flips,
                     win_form_err_range=float(ferr[lo:e + 1, b].max() - ferr[lo:e + 1, b].min()))
        # exploratory: outcome under looser tolerances (exact; see module docstring)
        last = e - 1 if oc == "COLLISION" else e          # collision beats goal in the same period
        for gt in ALT_GOAL_TOL:
            for ft in ALT_FORM_TOL:
                ok = np.flatnonzero((cd[:last + 1] <= gt) & (fe[:last + 1] <= ft)) if last >= 0 else np.array([], int)
                r[f"alt_g{gt:g}_f{ft:g}"] = int(ok[0]) if len(ok) else -1        # first period meeting it, -1 never
        assert (r["alt_g0.1_f0.1"] >= 0) == (oc == "GOAL") and (oc != "GOAL" or r["alt_g0.1_f0.1"] == e)
        rows.append(r)
    run = dict(policy_id=pol_rec["policy_id"], density=K, role=DENSITIES[K], layout_seed=cfg.validation_seed(K),
               layout_hash=layout_hash(lay), n=B, batch_periods=S, wall_s=round(wall, 2), actor_digest=pol.digest,
               env_fingerprint=cfg.env_fingerprint())
    return run, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/phase1/eval")
    a = ap.parse_args()
    from cbf_study.evaluate import write_rows
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(p, K) for p in sel["policies"] for K in DENSITIES]
    t0 = time.time()
    runs, rows = [], []
    with ProcessPoolExecutor(a.workers) as ex:
        for run, rs in ex.map(run_task, tasks):
            runs.append(run)
            rows += rs
            g = sum(r["outcome"] == "GOAL" for r in rs)
            c = sum(r["outcome"] == "COLLISION" for r in rs)
            print(f"{run['policy_id']} K{run['density']}: GOAL {g}/200 COLL {c}/200 periods {run['batch_periods']} "
                  f"{run['wall_s']} s", flush=True)
    write_rows(out / "trials.csv", rows)
    write_rows(out / "runs.csv", runs)
    meta = dict(script="scripts/phase1_eval.py", filter="OFF", split="validation", trials_per_density=200,
                densities=DENSITIES, alt_goal_tol=ALT_GOAL_TOL, alt_form_tol=ALT_FORM_TOL, near_goal_m=NEAR_GOAL,
                window=WINDOW, workers=a.workers, wall_s=round(time.time() - t0, 1),
                env_fingerprint=sel["env_fingerprint"])
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
