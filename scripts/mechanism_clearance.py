#!/usr/bin/env python3
"""EXPLORATORY, POST HOC: clearance analysis and filter-C command checks on the saved final-test
logs (commits e23f8c3, 41ae3d4). No simulation, no environment stepping, no QP solve.

Groups (pairs by layout, filter off vs filter on at alpha* = 10, K = 1-3, selected by observed outcome):
  A  = off GOAL and on TIMEOUT classified 'filter-induced stall' by the existing rule
       (results/final_test/mechanism/timeout_classes.csv); subsets A_obst (obstacle or
       obstacle+neighbour) and A_nb (neighbour only)
  A' = off GOAL and on not GOAL (any reason)
  B  = off GOAL and on GOAL
Filter C = the unchanged CBF-QP rows at alpha = 10 (cbf_qp construction: margins, equal split,
sensing / communication masks), rows evaluated only; a row is violated when (b - a.u)/|a| > solver_tol
(qp2d normalisation; a degenerate row |a| < 1e-12 is violated when b > solver_tol).
Executed periods only: pre-step state k and applied command k for k = 0..T-1; state samples t = 0..T
(initial and terminal included, frozen post-termination samples excluded). All clearances are
control-instant samples, not continuous-path minima.

Writes results/final_test/mechanism/{clearance_trajectories.csv, clearance_summary.csv, clearance_summary.json}."""
import csv, gzip, json, sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbf_study.bmatch import barrier_rows, rows_violated     # noqa: E402
from cbf_study.cbf_qp import barrier_point, margin_obstacle, margin_robot, rot_jacobian   # noqa: E402
from cbf_study.config import StudyConfig                     # noqa: E402
from cbf_study.evaluate import write_rows                    # noqa: E402

FT = ROOT / "results" / "final_test"
OUT = FT / "mechanism"
LABEL = "EXPLORATORY, POST HOC"
STALL_DIST = 0.05


def outcomes():
    o = {}
    for p in sorted((FT / "cells").glob("*.csv.gz")):
        for r in csv.DictReader(gzip.open(p, "rt")):
            o[(r["policy_id"], int(r["density"]), int(r["trial"]), r["condition"])] = r["outcome"]
    return o


def groups(out):
    cls = {(r["policy_id"], int(r["density"]), int(r["trial"])): r["category"]
           for r in csv.DictReader(open(OUT / "timeout_classes.csv")) if r["condition"] == "on"}
    g = {}
    for (pid, K, t, c), oc in out.items():
        if c != "off" or oc != "GOAL":
            continue
        on = out[(pid, K, t, "on")]
        if on == "GOAL":
            g[(pid, K, t)] = ("B", "GOAL")
        else:
            sub = cls.get((pid, K, t), on) if on == "TIMEOUT" else on
            g[(pid, K, t)] = ("A'", sub)
    return g


def stall_onset(cen, T):
    """first executed state from which the centroid stays within STALL_DIST of its final position"""
    d = np.linalg.norm(cen[:T + 1] - cen[T], axis=1)
    far = np.flatnonzero(d > STALL_DIST)
    return int(far[-1] + 1) if len(far) else 0


def main():
    cfg = StudyConfig.load()
    fc = cfg.filter_config(10.0)
    a, tol, dt = fc.alpha, fc.solver_tol, fc.dt
    R, Ro, ell = fc.robot_r, fc.obst_r, fc.ell
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())
    pols = [p["policy_id"] for p in sel["policies"]]
    out = outcomes()
    grp = groups(out)
    traj_rows = []
    for pid in pols:
        for K in (1, 2, 3):
            lay = np.load(FT / "layouts" / f"{pid}_K{K}.npz")
            obst_all = lay["obst_xy"]
            z = dict(np.load(FT / "steps" / f"{pid}_K{K}_off.npz"))      # decompress each array once
            zo = dict(np.load(FT / "steps" / f"{pid}_K{K}_on.npz"))
            T, po, co = z["executed_periods"], z["pose_offsets"], z["cmd_offsets"]
            assert (z["attempted_periods"] == T).all()
            poses = z["poses"].astype(np.float64)
            for t in [t for t in range(len(T)) if (pid, K, t) in grp]:
                g, sub = grp[(pid, K, t)]
                Tt = int(T[t])
                S = poses[po[t]:po[t] + Tt + 1]                         # states t = 0..T
                U = z["applied"][co[t]:co[t] + Tt].astype(np.float64)    # executed commands k = 0..T-1
                assert np.isfinite(U).all()
                obst = np.repeat(obst_all[t][None], Tt + 1, axis=0)
                pb = barrier_point(S, ell)                                # [T+1,N,2]
                # ---- physical clearances and cover barriers on all states 0..T ----
                gap_o = (np.linalg.norm(S[:, :, None, :2] - obst[:, None], axis=3) - R - Ro) if K else np.full((Tt + 1, 3, 1), np.inf)
                hob = (np.linalg.norm(pb[:, :, None] - obst[:, None], axis=3) ** 2 - fc.d_obst ** 2) if K else np.full((Tt + 1, 3, 1), np.inf)
                i, j = np.triu_indices(3, 1)
                gap_r = np.linalg.norm(S[:, i, :2] - S[:, j, :2], axis=2) - 2 * R
                hrr = np.linalg.norm(pb[:, i] - pb[:, j], axis=2) ** 2 - fc.d_rr ** 2
                row = dict(label=LABEL, policy_id=pid, seed=int(pid[1]), density=K, trial=t, group=g, on_class=sub,
                           in_A=int(sub.startswith("filter-induced")),
                           in_A_obst=int(sub.startswith("filter-induced") and "obstacle" in sub),
                           in_A_nb=int(sub == "filter-induced stall: neighbour"),
                           off_T=Tt, min_gap_obst=float(gap_o.min()), min_gap_rr=float(gap_r.min()),
                           h_obst_neg_initial=int((hob[0] < 0).any()), h_obst_neg_later=int((hob[1:] < 0).any() and not (hob[0] < 0).any()),
                           h_obst_neg_any=int((hob < 0).any()),
                           h_rr_neg_initial=int((hrr[0] < 0).any()), h_rr_neg_later=int((hrr[1:] < 0).any() and not (hrr[0] < 0).any()),
                           h_rr_neg_any=int((hrr < 0).any()))
                if Tt == 0:
                    traj_rows.append(row)
                    continue
                # ---- filter-C rows at pre-step states k = 0..T-1 with the applied command ----
                P = S[:Tt]
                Ob = obst[:Tt]
                A_, b_, m_ = barrier_rows(fc, P, Ob)
                p = pb[:Tt]
                J = rot_jacobian(P[..., 2], ell)
                nrm = np.linalg.norm(A_, axis=3)
                au = np.einsum("knrd,knd->knr", A_, U)
                res = (au - b_) / np.where(nrm < 1e-12, 1.0, nrm)              # normalised residual; < -tol = violated
                viol = m_ & (((nrm >= 1e-12) & (res < -tol)) | ((nrm < 1e-12) & (b_ > tol)))
                assert (viol.any(axis=2) == rows_violated(U, A_, b_, m_, tol)).all()
                vO, vR = viol[:, :, :K], viol[:, :, K:]
                # barrier values per row
                hO = (np.linalg.norm(p[:, :, None] - Ob[:, None], axis=3) ** 2 - fc.d_obst ** 2) if K else np.zeros((Tt, 3, 0))
                dR = np.linalg.norm(p[:, :, None] - p[:, None], axis=3)
                hR = dR ** 2 - fc.d_rr ** 2
                # unmarginated residuals (normalised by |g| like the rows)
                gO_u = au[:, :, :K]
                unO = (gO_u + a * hO) / np.where(nrm[:, :, :K] < 1e-12, 1.0, nrm[:, :, :K])
                gR_u = au[:, :, K:]                                            # g_ij . u_i, [k, i, j]
                unR_local = (gR_u + 0.5 * a * hR) / np.where(nrm[:, :, K:] < 1e-12, 1.0, nrm[:, :, K:])
                joint = gR_u + np.swapaxes(gR_u, 1, 2) + a * hR                 # g_i u_i + g_j u_j + alpha h
                catO = dict(a=vO & (hO < 0), b=vO & (hO >= 0) & (unO >= -tol), c=vO & (hO >= 0) & (unO < -tol))
                catR = dict(a=vR & (hR < 0), b=vR & (hR >= 0) & (unR_local >= -tol), c=vR & (hR >= 0) & (unR_local < -tol))
                for nm, cat in (("obst", catO), ("rr", catR)):
                    v = vO if nm == "obst" else vR
                    row[f"C_{nm}_viol_any"] = int(v.any())
                    for c_, mask in cat.items():
                        row[f"C_{nm}_cat_{c_}"] = int(mask.any())
                    if v.any():
                        k0 = int(np.flatnonzero(v.any(axis=(1, 2)))[0])
                        rr_, cc_ = np.argwhere(v[k0])[np.argmin(res[k0, :, (slice(0, K) if nm == 'obst' else slice(K, None))][v[k0]])]
                        col = cc_ if nm == "obst" else K + cc_
                        hval = hO[k0, rr_, cc_] if nm == "obst" else hR[k0, rr_, cc_]
                        row.update({f"C_{nm}_first_period": k0, f"C_{nm}_first_time_s": k0 * dt,
                                    f"C_{nm}_first_h": float(hval), f"C_{nm}_first_residual": float(res[k0, rr_, col]),
                                    f"C_{nm}_first_unmarg": float((unO if nm == "obst" else unR_local)[k0, rr_, cc_]),
                                    f"C_{nm}_first_cat": "a" if hval < 0 else ("b" if (unO if nm == "obst" else unR_local)[k0, rr_, cc_] >= -tol else "c"),
                                    f"C_{nm}_viol_at_initial": int(k0 == 0),
                                    f"C_{nm}_first_gap_obst": float(gap_o[k0].min()), f"C_{nm}_first_gap_rr": float(gap_r[k0].min())})
                        if nm == "rr":
                            row["C_rr_first_joint_satisfied"] = int(joint[k0, rr_, cc_] >= -tol)
                if vR.any():
                    row["C_rr_any_viol_joint_satisfied"] = int((vR & (joint >= -tol)).any())
                    row["C_rr_any_viol_joint_violated"] = int((vR & (joint < -tol)).any())
                # ---- zero-command check: states with every h >= 0 where u = 0 violates an enabled row ----
                safe = (hO >= 0).all(axis=(1, 2)) & (hR + np.eye(3)[None] * 1e9 >= 0).all(axis=(1, 2))
                zviol = m_ & (b_ > tol * np.where(nrm < 1e-12, 1.0, nrm))
                zO, zR = zviol[:, :, :K].any(axis=(1, 2)) & safe, zviol[:, :, K:].any(axis=(1, 2)) & safe
                row.update(zero_cmd_inadmissible_obst=int(zO.any()), zero_cmd_inadmissible_rr=int(zR.any()),
                           zero_cmd_first_period=int(np.flatnonzero(zO | zR)[0]) if (zO | zR).any() else -1,
                           zero_cmd_share_of_periods=float((zO | zR).mean()))
                # ---- filter-on trajectory: clearance at stall onset (group A) and minima ----
                Ton = int(zo["executed_periods"][t])
                So = zo["poses"][zo["pose_offsets"][t]:zo["pose_offsets"][t] + Ton + 1].astype(np.float64)
                go = (np.linalg.norm(So[:, :, None, :2] - obst_all[t][None, None], axis=3) - R - Ro) if K else np.full((Ton + 1, 3, 1), np.inf)
                gr = np.linalg.norm(So[:, i, :2] - So[:, j, :2], axis=2) - 2 * R
                row.update(on_T=Ton, on_min_gap_obst=float(go.min()), on_min_gap_rr=float(gr.min()))
                if row["in_A"]:
                    s0 = stall_onset(So[..., :2].mean(axis=1), Ton)
                    row.update(on_stall_onset=s0, on_gap_obst_at_stall=float(go[s0].min()), on_gap_rr_at_stall=float(gr[s0].min()))
                traj_rows.append(row)
        print(pid, flush=True)
    keys = sorted({k for r in traj_rows for k in r}, key=lambda k: list(traj_rows[0]).index(k) if k in traj_rows[0] else 999)
    write_rows(OUT / "clearance_trajectories.csv", [{k: r.get(k, "") for k in keys} for r in traj_rows])
    summarise(traj_rows, pols)


def summarise(rows, pols):
    G = {"A": lambda r: r["in_A"] == 1, "A_obst": lambda r: r["in_A_obst"] == 1, "A_nb": lambda r: r["in_A_nb"] == 1,
         "A'": lambda r: r["group"] == "A'", "B": lambda r: r["group"] == "B"}
    flags = ["h_obst_neg_any", "h_obst_neg_initial", "h_obst_neg_later", "h_rr_neg_any", "h_rr_neg_initial", "h_rr_neg_later",
             "C_obst_viol_any", "C_obst_cat_a", "C_obst_cat_b", "C_obst_cat_c", "C_obst_viol_at_initial",
             "C_rr_viol_any", "C_rr_cat_a", "C_rr_cat_b", "C_rr_cat_c", "C_rr_viol_at_initial",
             "C_rr_first_joint_satisfied", "C_rr_any_viol_joint_satisfied", "C_rr_any_viol_joint_violated",
             "zero_cmd_inadmissible_obst", "zero_cmd_inadmissible_rr"]
    quants = ["min_gap_obst", "min_gap_rr", "C_obst_first_period", "C_obst_first_h", "C_obst_first_residual", "C_obst_first_gap_obst",
              "C_rr_first_period", "C_rr_first_h", "C_rr_first_residual", "C_rr_first_gap_rr", "zero_cmd_first_period",
              "on_min_gap_obst", "on_min_gap_rr", "on_gap_obst_at_stall", "on_gap_rr_at_stall"]
    thr = {"min_gap_obst": (0.0, 0.05, 0.10), "min_gap_rr": (0.0, 0.05, 0.10), "on_min_gap_obst": (0.0, 0.05, 0.10),
           "on_min_gap_rr": (0.0, 0.05, 0.10), "on_gap_obst_at_stall": (0.05, 0.10, 0.20), "on_gap_rr_at_stall": (0.05, 0.10, 0.20)}
    out = []
    scopes = [("pooled", "all", lambda r: True)] + [("density", str(K), (lambda K: lambda r: r["density"] == K)(K)) for K in (1, 2, 3)] \
        + [("actor", p, (lambda p: lambda r: r["policy_id"] == p)(p)) for p in pols]
    for gname, gf in G.items():
        for sc, sid, sf in scopes:
            rs = [r for r in rows if gf(r) and sf(r)]
            if not rs:
                continue
            base = dict(label=LABEL, group=gname, scope=sc, scope_id=sid, n_trajectories=len(rs))
            for f in flags:
                vals = [r[f] for r in rs if f in r and r[f] != ""]
                den = len(rs) if not f.startswith("C_rr_first_joint") and not f.startswith("C_rr_any_viol") else sum(r.get("C_rr_viol_any", 0) for r in rs)
                out.append(dict(base, quantity=f, kind="share", numerator=int(sum(vals)), denominator=den,
                                share=(sum(vals) / den) if den else float("nan")))
            for q in quants:
                v = np.array([r[q] for r in rs if r.get(q, "") != "" and np.isfinite(r[q]) and not (q.endswith("period") and r[q] < 0)], float)
                if not len(v):
                    continue
                d = dict(base, quantity=q, kind="distribution", n=len(v), min=float(v.min()),
                         **{f"p{pp}": float(np.percentile(v, pp)) for pp in (10, 25, 50, 75, 90)})
                for t_ in thr.get(q, ()):
                    d[f"share_below_{t_:g}"] = float((v < t_).mean())
                out.append(d)
    keys = sorted({k for r in out for k in r}, key=lambda k: list(out[0]).index(k) if k in out[0] else 999)
    write_rows(OUT / "clearance_summary.csv", [{k: r.get(k, "") for k in keys} for r in out])
    counts = {g: sum(1 for r in rows if f(r)) for g, f in G.items()}
    breakdown = defaultdict(int)
    for r in rows:
        if r["group"] == "A'":
            breakdown[r["on_class"]] += 1
    (OUT / "clearance_summary.json").write_text(json.dumps(dict(label=LABEL, group_counts=counts,
                                                                 A_prime_breakdown=dict(breakdown)), indent=1))
    print(json.dumps(dict(group_counts=counts, A_prime_breakdown=dict(breakdown)), indent=1))


if __name__ == "__main__":
    main()
