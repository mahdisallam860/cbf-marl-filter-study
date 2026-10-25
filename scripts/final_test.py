#!/usr/bin/env python3
"""Locked FINAL TEST (docs/PREFILTER_VALIDATION_PROTOCOL_LOCKED_2026-10-25.md, D2 / D6) at
alpha* = 10 (results/validation_sweep/alpha_star.json, commit 4fffe62).

Stage 1 (once): generate the reserved layouts, seed = 4,000,000 + 1,000 p + K, 1,000 per actor and
K = 1, 2, 3, save the arrays and their SHA-256 under results/final_test/layouts/. A saved set is never
regenerated (an existing file is only re-hashed). The 5,000,000 block is never used (asserted).
Stage 2: per (actor, K), on the saved layouts: filter off, filter on at alpha*, B-match at alpha*,
through the unchanged protocol loop. Per trial: the scalar row (same fields as the development sweep).
Per step, every trial: realised poses (t = 0..T), raw / clipped / applied commands, filter interventions,
QP status and infeasible flags, B-match brake activations, pre-step barrier minima. One file per cell
and condition; resumable.

Writes results/final_test/{layouts/, cells/, steps/, run.log, final_meta.json}."""
import argparse, csv, gzip, hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "final_test"
K_PRIMARY = (1, 2, 3)
N_LAYOUTS = 1000
ALPHA_STAR = 10.0


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def generate_layouts(cfg, sel):
    from cbf_study.scenario import sample_layout
    res = json.loads((ROOT / "docs/final_test_seed_reservation.json").read_text())
    lo, hi = res["reserved_range"]
    m_lo, m_hi = res["method_study_block"]["range"]
    seeds = {(s["policy_id"], s["density"]): s["seed"] for s in res["seeds"]}
    d = OUT / "layouts"
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for p, pol in enumerate(sel["policies"]):
        for K in K_PRIMARY:
            seed = 4_000_000 + 1000 * p + K
            assert seed == seeds[(pol["policy_id"], K)] and lo <= seed <= hi and not m_lo <= seed <= m_hi
            f = d / f"{pol['policy_id']}_K{K}.npz"
            if f.exists():
                z = np.load(f)
                arrs = {k: z[k] for k in ("init_poses", "obst_xy", "goal", "slots")}
                generated = "loaded (generated earlier in this run)"
            else:
                lay = sample_layout(seed, N_LAYOUTS, K, cfg.layout_filter_config(), cfg.scenario_config())
                arrs = dict(init_poses=lay.init_poses, obst_xy=lay.obst_xy, goal=lay.goal, slots=lay.slots)
                np.savez(f, **arrs)
                generated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            rows.append(dict(policy_index=p, policy_id=pol["policy_id"], density=K, seed=seed, n=N_LAYOUTS,
                             **{f"sha256_{k}": sha(v) for k, v in arrs.items()},
                             sha256_file=hashlib.sha256(f.read_bytes()).hexdigest(), generated_utc=generated))
    return rows


def ragged(x, counts):
    """x [S,B,...] -> concatenation over trials b of x[:counts[b], b]."""
    return np.concatenate([x[:c, b] for b, c in enumerate(counts)], axis=0)


def run_cell(args):
    pol_rec, K = args
    import torch
    torch.set_num_threads(1)
    from cbf_study.bmatch import barrier_rows, rows_violated, run_bmatch
    from cbf_study.cbf_qp import barrier_point
    from cbf_study.config import StudyConfig, check_paired
    from cbf_study.evaluate import vmas_factory
    from cbf_study.policies import FrozenMAPPOPolicy
    from cbf_study.protocol import Outcome, run_trials
    from cbf_study.scenario import Layout
    from cbf_study.trial_metrics import (RecordingEnv, centroid_distance_series, common_prefix, formation_series,
                                         full_trajectory_mean, reached_goal_region, sampled_barrier_minima)
    cfg = StudyConfig.load()
    pol = FrozenMAPPOPolicy(ROOT / pol_rec["ckpt"], cfg)
    assert pol.digest == pol_rec["actor_param_digest"]
    z = np.load(OUT / "layouts" / f"{pol_rec['policy_id']}_K{K}.npz")
    lay = Layout(z["init_poses"], z["obst_xy"], z["goal"], z["slots"])
    lh = hashlib.sha256(b"".join(np.ascontiguousarray(a).tobytes() for a in (lay.init_poses, lay.obst_xy, lay.goal, lay.slots))).hexdigest()[:16]
    pc = cfg.protocol_config()
    fc_off = cfg.filter_config(cfg.alphas[0])
    fc_star = cfg.filter_config(ALPHA_STAR)
    T, (B, N) = pc.max_steps, lay.init_poses.shape[:2]
    steps_dir = OUT / "steps"

    def run(cond):
        fc = fc_off if cond == "off" else fc_star
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
        L = res.step_log
        att = L["active"].sum(axis=0).astype(int)                 # attempted periods (incl. an infeasible attempt)
        assert ((att == Ta) | ((att == Ta + 1) & (res.outcome == Outcome.INFEASIBLE))).all()
        # ---- per-step record, every trial (ragged, lossless at the logged precision) ----
        brake = L.get("brake_active", np.zeros_like(L["intervened"]))
        np.savez_compressed(
            steps_dir / f"{pol_rec['policy_id']}_K{K}_{cond}.npz",
            condition=cond, alpha=np.nan if cond == "off" else ALPHA_STAR, layout_hash=lh,
            executed_periods=Ta, attempted_periods=att, outcome=res.outcome.astype(np.int8),
            pose_offsets=np.r_[0, np.cumsum(Ta + 1)], cmd_offsets=np.r_[0, np.cumsum(att)],
            poses=ragged(traj, Ta + 1).astype(np.float32),                         # VMAS state is float32
            raw=ragged(L["raw"], att).astype(np.float32),                          # actor output (float32 network)
            clipped=ragged(L["clipped"], att).astype(np.float32),
            applied=ragged(L["applied"], att),                                     # float64 (QP output); NaN = not applied
            intervened=ragged(L["intervened"], att), brake_active=ragged(brake, att),
            status=ragged(L["status"], att).astype(np.int8), infeasible=ragged(L["infeasible"], att),
            h_obst_min_prestep=ragged(L["h_obst_min"], att).astype(np.float32),
            h_rr_min_prestep=ragged(L["h_rr_min"], att).astype(np.float32))
        e = formation_series(traj, lay.slots)
        cd = centroid_distance_series(traj, lay.goal)
        hos, hrs = sampled_barrier_minima(traj, Ta, lay.obst_xy, fc)
        p = barrier_point(traj, fc.ell)
        i, j = np.triu_indices(N, 1)
        sep = np.linalg.norm(p[:, :, i] - p[:, :, j], axis=3).min(axis=2)
        sep_min = np.array([sep[:Ta[b] + 1, b].min() for b in range(B)])
        rr_frac = np.full(B, np.nan)
        if cond != "off":
            hits = np.zeros(B)
            Kn = lay.obst_xy.shape[1]
            for k in range(int(Ta.max())):
                ex = k < Ta
                A, bb, m = barrier_rows(fc, traj[k], lay.obst_xy)
                v = rows_violated(np.nan_to_num(L["clipped"][k]), A[:, :, Kn:], bb[:, :, Kn:], m[:, :, Kn:], fc.solver_tol)
                hits += np.where(ex, v.sum(axis=1), 0)
            rr_frac = np.where(Ta > 0, hits / np.maximum(Ta * N, 1), np.nan)
        return dict(cond=cond, alpha=None if cond == "off" else ALPHA_STAR, res=res, e=e, T=Ta, wall=wall, hos=hos, hrs=hrs,
                    sep_min=sep_min, sep0=sep[0], rr_frac=rr_frac, ftm=full_trajectory_mean(e, Ta),
                    rgr=reached_goal_region(cd, Ta, res.outcome, 0.30))

    runs = [run("off"), run("on"), run("bmatch")]
    off, on = runs[0], runs[1]
    rows = []
    for r in runs:
        res = r["res"]
        d_off, L_off, why_off = common_prefix(off["e"], off["T"], r["e"], r["T"])
        if r["cond"] == "bmatch":
            d_ob, L_ob, why_ob = common_prefix(r["e"], r["T"], on["e"], on["T"])
        else:
            d_ob, L_ob, why_ob = np.full(B, np.nan), np.zeros(B, int), np.full(B, "n/a", dtype=object)
        for b in range(B):
            oc = Outcome(res.outcome[b]).name
            rows.append(dict(
                policy_id=pol_rec["policy_id"], seed=pol_rec["seed"], phase=pol_rec["phase"], density=K, role="primary",
                condition=r["cond"], alpha="" if r["alpha"] is None else r["alpha"], trial=b,
                layout_seed=4_000_000 + 1000 * pol_rec["index"] + K, layout_hash=lh, outcome=oc,
                collision_kind={0: "none", 1: "obstacle", 2: "robot"}[int(res.collision_kind[b])],
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
    meta = dict(policy_id=pol_rec["policy_id"], density=K, layout_hash=lh, actor_digest=pol.digest,
                walls={r["cond"]: round(r["wall"], 2) for r in runs})
    return pol_rec["policy_id"], K, rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    from cbf_study.config import StudyConfig
    cfg = StudyConfig.load()
    assert cfg.env_fingerprint() == "c60352aee3f11ed8"
    star = json.loads((ROOT / "results/validation_sweep/alpha_star.json").read_text())
    assert star["alpha_star"] == ALPHA_STAR
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())
    for i, p in enumerate(sel["policies"]):
        p["index"] = i
    (OUT / "cells").mkdir(parents=True, exist_ok=True)
    (OUT / "steps").mkdir(parents=True, exist_ok=True)
    lrows = generate_layouts(cfg, sel)
    from cbf_study.evaluate import write_rows
    if not (OUT / "layouts.csv").exists():
        write_rows(OUT / "layouts.csv", lrows)
    else:                                                     # a resumed run must see byte-identical layouts
        old = {(r["policy_id"], r["density"]): r for r in csv.DictReader(open(OUT / "layouts.csv"))}
        for r in lrows:
            assert old[(r["policy_id"], str(r["density"]))]["sha256_init_poses"] == r["sha256_init_poses"]
    tasks = [(p, K) for K in K_PRIMARY for p in sel["policies"] if not (OUT / "cells" / f"{p['policy_id']}_K{K}.csv.gz").exists()]
    print(f"{len(tasks)} cells to run", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futs = [ex.submit(run_cell, t) for t in tasks]
        for f in as_completed(futs):
            pid, K, rows, meta = f.result()
            tmp = OUT / "cells" / f"{pid}_K{K}.csv.gz.tmp"
            with gzip.open(tmp, "wt", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows({k: (f"{v:.7g}" if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
            (OUT / "cells" / f"{pid}_K{K}.meta.json").write_text(json.dumps(meta, indent=1))
            tmp.rename(OUT / "cells" / f"{pid}_K{K}.csv.gz")
            print(f"{pid} K{K} done {meta['walls']} elapsed {time.time() - t0:.0f} s", flush=True)
    (OUT / "final_meta.json").write_text(json.dumps(dict(
        protocol="docs/PREFILTER_VALIDATION_PROTOCOL_LOCKED_2026-10-25.md", protocol_commit="9baf6d7",
        alpha_star=ALPHA_STAR, alpha_star_commit="4fffe62", env_fingerprint=cfg.env_fingerprint(), densities=K_PRIMARY,
        layouts_per_actor_and_density=N_LAYOUTS, conditions=["off", "on", "bmatch"], workers=a.workers,
        wall_s_this_invocation=round(time.time() - t0, 1)), indent=1))


if __name__ == "__main__":
    main()
