#!/usr/bin/env python3
"""EXPLORATORY, POST HOC infeasibility audit of the final test (saved logs and layouts only; no
simulation, no environment stepping, no new layouts; outcomes are not relabelled).

(a) For each of the 433 filter-on INFEASIBLE trials: the saved pre-step state of the attempted
    (unexecuted) period and the saved raw actor command. Reproduction: the unchanged CBFQPFilter
    (alpha* = 10) is called once on these saved values (its own float64 promotion); the verdict is
    only compared with the logged one. Rows: cbf_study.bmatch.barrier_rows (identical construction),
    in float64, and an independent float32-arithmetic evaluation of the same row formulas.
    Normalised slack of a row at command u: (a.u - b)/|a| (negative = violated).
    Stand-off of a row: h* = (row margin)/(row gain): obstacle delta_o/alpha, pair (delta_ij/2)/(alpha/2).
    Declared near-boundary diagnostic (fixed before computing): |zero-command slack of the binding
    row| <= 1e-6 AND |h - h*| <= 1e-3 m^2 for that row, binding row = the enabled row with the most
    negative zero-command slack of an infeasible robot. Reported as 'consistent with numerical
    sensitivity near the boundary', never as proof of the cause.
(b) Initial states (A5 check): all 27,000 final-test trials (initial poses from the saved per-step
    logs) and the saved validation layout arrays audit/layouts_HEAD/validation_K{0,1,2,3,5,8}_n200.npz
    (K = 4 is not saved). Counts of trials with any negative cover barrier at t = 0, with body
    clearances.
Writes results/final_test/INFEASIBILITY_AUDIT.md and results/final_test/infeasibility/*.csv."""
import csv, gzip, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbf_study.bmatch import barrier_rows                    # noqa: E402
from cbf_study.cbf_qp import CBFQPFilter, barrier_values, margin_obstacle, margin_robot   # noqa: E402
from cbf_study.config import StudyConfig                     # noqa: E402
from cbf_study.evaluate import write_rows                    # noqa: E402

FT = ROOT / "results" / "final_test"
OUT = FT / "infeasibility"
LABEL = "EXPLORATORY, POST HOC"
SLACK_TOL, H_TOL = 1e-6, 1e-3


def rows32(fc, pose, obst, u):
    """Float32-arithmetic evaluation of the same row formulas for one robot set. Returns
    (zero-command slack [N,R], command slack [N,R], mask [N,R]) in float32."""
    f = np.float32
    pose, obst, u = pose.astype(f), obst.astype(f), u.astype(f)
    ell, a = f(fc.ell), f(fc.alpha)
    c, s = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    p = np.stack([pose[:, 0] + ell * c, pose[:, 1] + ell * s], axis=1)
    J = np.stack([np.stack([c, -ell * s], 1), np.stack([s, ell * c], 1)], 1)       # [N,2,2]
    vp, ap, dt = f(fc.vp), f(fc.ap), f(fc.dt)
    A, b, m = [], [], []
    if len(obst):
        q = p[:, None] - obst[None]
        d = np.linalg.norm(q, axis=2)
        A.append(f(2) * np.einsum("nji,nkj->nki", J, q))
        b.append(-a * (d * d - f(fc.d_obst) ** 2) + f(0.5) * dt * (f(2) * vp ** 2 + f(2) * (d + vp * dt) * ap))
        m.append(d <= f(fc.sensing_range))
    q = p[:, None] - p[None]
    d = np.linalg.norm(q, axis=2)
    A.append(f(2) * np.einsum("nji,nkj->nki", J, q))
    b.append(f(-0.5) * a * (d * d - f(fc.d_rr) ** 2) + f(0.5) * f(0.5) * dt * (f(8) * vp ** 2 + f(4) * (d + f(2) * vp * dt) * ap))
    m.append((d <= f(fc.comm_range)) & ~np.eye(len(pose), dtype=bool))
    A, b, m = np.concatenate(A, 1), np.concatenate(b, 1), np.concatenate(m, 1)
    nrm = np.linalg.norm(A, axis=2)
    safe = np.where(nrm < f(1e-12), f(1), nrm)
    return -b / safe, (np.einsum("nrd,nd->nr", A, u) - b) / safe, m


def part_a(cfg, fc, sel):
    rows_r, rows_t = [], []
    filt = CBFQPFilter(fc)
    for p in sel["policies"]:
        pid = p["policy_id"]
        for K in (1, 2, 3):
            obst_all = np.load(FT / "layouts" / f"{pid}_K{K}.npz")["obst_xy"]
            z = dict(np.load(FT / "steps" / f"{pid}_K{K}_on.npz"))
            T, att, po, co = z["executed_periods"], z["attempted_periods"], z["pose_offsets"], z["cmd_offsets"]
            for t in np.flatnonzero(z["outcome"] == 3):
                k = int(T[t])
                row = co[t] + att[t] - 1
                assert att[t] == k + 1
                pose32 = z["poses"][po[t] + k]                                 # saved (float32) pre-step state
                raw32 = z["raw"][row]
                obst = obst_all[t]
                logged_inf = z["infeasible"][row]
                out = filt(pose32[None].astype(np.float64), raw32[None].astype(np.float64), obst[None])
                rep = out["infeasible"][0]
                reproduced = bool((rep == logged_inf).all() and rep.any())
                A, b, m = barrier_rows(fc, pose32[None].astype(np.float64), obst[None])
                A, b, m = A[0], b[0], m[0]
                nrm = np.linalg.norm(A, axis=2)
                safe = np.where(nrm < 1e-12, 1.0, nrm)
                clip = np.clip(raw32.astype(np.float64), [fc.v_min, -fc.w_max], [fc.v_max, fc.w_max])
                s0 = -b / safe
                sn = (np.einsum("nrd,nd->nr", A, clip) - b) / safe
                s0_32, sn_32, m32 = rows32(fc, pose32, obst, clip)
                h_o, h_r = barrier_values(fc, pose32[None].astype(np.float64), obst[None])
                hrow = np.concatenate([h_o[0], h_r[0]], axis=1)                 # [N, K+N]
                pb = pose32.astype(np.float64)[:, :2] + fc.ell * np.stack([np.cos(pose32[:, 2]), np.sin(pose32[:, 2])], 1)
                d_o = np.linalg.norm(pb[:, None] - obst[None], axis=2)
                d_r = np.linalg.norm(pb[:, None] - pb[None], axis=2)
                hstar = np.concatenate([margin_obstacle(fc, d_o) / fc.alpha, margin_robot(fc, d_r) / fc.alpha], axis=1)
                trial_min = []
                for i in np.flatnonzero(logged_inf):
                    en = m[i]
                    sl = np.where(en, s0[i], np.inf)
                    jb = int(np.argmin(sl))
                    near = bool(abs(s0[i, jb]) <= SLACK_TOL and abs(hrow[i, jb] - hstar[i, jb]) <= H_TOL)
                    trial_min.append(s0[i, jb])
                    rows_r.append(dict(label=LABEL, policy_id=pid, seed=p["seed"], density=K, trial=int(t), robot=int(i), period=k,
                                       reproduced=int(reproduced), enabled_obstacle_rows=int(en[:K].sum()), enabled_neighbour_rows=int(en[K:].sum()),
                                       binding_row="obstacle" if jb < K else "neighbour", binding_zero_slack_f64=float(s0[i, jb]),
                                       binding_zero_slack_f32=float(s0_32[i, jb]), binding_nominal_slack_f64=float(sn[i, jb]),
                                       binding_h=float(hrow[i, jb]), binding_hstar=float(hstar[i, jb]),
                                       binding_h_minus_hstar=float(hrow[i, jb] - hstar[i, jb]),
                                       min_zero_slack_all_rows=float(sl.min()), n_rows_zero_violated=int((en & (s0[i] < -fc.solver_tol)).sum()),
                                       min_nominal_slack_all_rows=float(np.where(en, sn[i], np.inf).min()),
                                       zero_slacks_enabled=";".join(f"{x:.3e}" for x in s0[i][en]),
                                       nominal_slacks_enabled=";".join(f"{x:.3e}" for x in sn[i][en]),
                                       h_enabled=";".join(f"{x:.5f}" for x in hrow[i][en]),
                                       hstar_enabled=";".join(f"{x:.5f}" for x in hstar[i][en]),
                                       near_boundary_diagnostic=int(near) if reproduced else ""))
                rows_t.append(dict(label=LABEL, policy_id=pid, seed=p["seed"], density=K, trial=int(t), period=k, reproduced=int(reproduced),
                                   n_infeasible_robots=int(logged_inf.sum()), min_zero_slack=float(min(trial_min))))
    return rows_r, rows_t


def initial_states(cfg, fc, sel):
    R, Ro = fc.robot_r, fc.obst_r
    out = []
    def check(src, pid, K, init, obst):
        h_o, h_r = barrier_values(fc, init, obst)
        n = init.shape[0]
        for b in np.flatnonzero((h_o.reshape(n, -1) < 0).any(1) | (h_r.reshape(n, -1) < 0).any(1)):
            go = np.linalg.norm(init[b, :, None, :2] - obst[b][None], axis=2) - R - Ro if obst.shape[1] else np.array([[np.inf]])
            gr = min(np.linalg.norm(init[b, i, :2] - init[b, j, :2]) - 2 * R for i in range(3) for j in range(i + 1, 3))
            out.append(dict(label=LABEL, source=src, policy_id=pid, density=K, trial=int(b), min_h_obst=float(h_o[b].min()) if h_o.size else np.nan,
                            min_h_rr=float(h_r[b].min()), min_body_gap_obst=float(go.min()), min_body_gap_rr=float(gr),
                            obst_to_start_centroid=float(np.linalg.norm(obst[b] - np.array(cfg["scenario.start_centroid"]), axis=1).min()) if obst.shape[1] else np.nan))
        return n
    totals = {}
    for p in sel["policies"]:
        for K in (1, 2, 3):
            z = dict(np.load(FT / "steps" / f"{p['policy_id']}_K{K}_off.npz"))
            init = z["poses"][z["pose_offsets"][:-1]].astype(np.float64)
            lay = np.load(FT / "layouts" / f"{p['policy_id']}_K{K}.npz")
            assert np.allclose(init, lay["init_poses"].astype(np.float32).astype(np.float64), atol=1e-6)
            totals[("final_test", K)] = totals.get(("final_test", K), 0) + check("final_test (saved per-step initial poses)", p["policy_id"], K, init, lay["obst_xy"])
    for K in (0, 1, 2, 3, 5, 8):
        z = np.load(ROOT / "audit" / "layouts_HEAD" / f"validation_K{K}_n200.npz")
        totals[("validation", K)] = check("validation (audit/layouts_HEAD)", "", K, z["init_poses"], z["obst_xy"])
    return out, totals


def main():
    cfg = StudyConfig.load()
    fc = cfg.filter_config(10.0)
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    rr, rt = part_a(cfg, fc, sel)
    write_rows(OUT / "infeasible_robots.csv", rr)
    write_rows(OUT / "infeasible_trials.csv", rt)
    init, totals = initial_states(cfg, fc, sel)
    write_rows(OUT / "initial_state_negative_barriers.csv", init if init else [dict(label=LABEL, note="none")])
    (OUT / "summary.json").write_text(json.dumps(dict(
        label=LABEL, infeasible_trials=len(rt), reproduced=sum(r["reproduced"] for r in rt),
        infeasible_robots=len(rr), near_boundary_reproduced_robots=sum(1 for r in rr if r["near_boundary_diagnostic"] == 1),
        reproduced_robots=sum(1 for r in rr if r["reproduced"]),
        initial_state_totals={f"{k[0]}_K{k[1]}": v for k, v in totals.items()}, initial_state_negative=len(init)), indent=1))
    print(json.load(open(OUT / "summary.json")))


if __name__ == "__main__":
    main()
