#!/usr/bin/env python3
"""EXPLORATORY, POST HOC: which constraints conflict in the infeasible filter-on QPs of the final test.

Saved data only: for each of the 433 INFEASIBLE trials, the saved pre-step state of the attempted
(unexecuted) period (results/final_test/steps/*_on.npz) and the saved layout. The filter's own row
construction (cbf_study.bmatch.barrier_rows, alpha* = 10) is evaluated at that state; no QP solver,
no CBFQPFilter call, no simulation, no environment stepping, no new layout, no outcome relabelled.

For one robot the constraint set is {a_r . u >= b_r for enabled rows r} together with the box
U = {|v| <= v_max, |w| <= w_max}. With normalised rows n_r = a_r/|a_r|, c_r = b_r/|a_r|, the
quantity
    t* = max_{u in U} min_r (n_r . u - c_r)          [m/s-equivalent normalised slack]
is the largest uniform slack any admissible command achieves; the set is empty iff t* < 0. t* is
computed exactly in float64 by enumerating the vertices of the 3-D linear programme (u, t)
(every triple of active planes among the row planes and the four box planes).

A minimal conflicting set is a subset S of enabled rows with t*_S < 0 (box kept) such that every
proper subset has t*_S >= 0. All minimal sets are enumerated (at most 5 enabled rows per robot).
Each set is also tested without the box (box widened to 1e3): if it then becomes feasible, the
command limits are part of the conflict.

Writes results/paper_revision/infeasibility_conflicts.csv and infeasibility_conflicts_summary.json."""
import csv, itertools, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbf_study.bmatch import barrier_rows                    # noqa: E402
from cbf_study.config import StudyConfig                     # noqa: E402

FT = ROOT / "results" / "final_test"
OUT = ROOT / "results" / "paper_revision"
LABEL = "exploratory, post hoc"


def tstar(N, c, vmax, wmax):
    """Exact max-min normalised slack over the box. N [k,2], c [k]."""
    # constraints g(x) <= h with x = (v, w, t)
    G = [np.array([-n[0], -n[1], 1.0]) for n in N]
    H = [-ci for ci in c]
    G += [np.array([1.0, 0, 0]), np.array([-1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, -1.0, 0])]
    H += [vmax, vmax, wmax, wmax]
    G, H = np.array(G), np.array(H)
    best = -np.inf
    for i, j, k in itertools.combinations(range(len(G)), 3):
        M = G[[i, j, k]]
        if abs(np.linalg.det(M)) < 1e-14:
            continue
        x = np.linalg.solve(M, H[[i, j, k]])
        if np.all(G @ x <= H + 1e-12 * (1 + np.abs(H))):
            best = max(best, x[2])
    return best


def main():
    cfg = StudyConfig.load()
    fc = cfg.filter_config(10.0)
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())
    rows = []
    for p in sel["policies"]:
        pid = p["policy_id"]
        for K in (1, 2, 3):
            obst_all = np.load(FT / "layouts" / f"{pid}_K{K}.npz")["obst_xy"]
            z = np.load(FT / "steps" / f"{pid}_K{K}_on.npz")
            T, att, po, co = z["executed_periods"], z["attempted_periods"], z["pose_offsets"], z["cmd_offsets"]
            out, poses, infz = z["outcome"], z["poses"], z["infeasible"]
            for t in np.flatnonzero(out == 3):
                k = int(T[t])
                pose = poses[po[t] + k].astype(np.float64)
                inf_rob = infz[co[t] + att[t] - 1]
                A, b, m = barrier_rows(fc, pose[None], obst_all[t][None])
                A, b, m = A[0], b[0], m[0]
                for i in np.flatnonzero(inf_rob):
                    en = np.flatnonzero(m[i])
                    nrm = np.linalg.norm(A[i, en], axis=1)
                    N, c = A[i, en] / nrm[:, None], b[i, en] / nrm
                    kinds = ["o" if r < K else "n" for r in en]
                    t_all = tstar(N, c, fc.v_max, fc.w_max)
                    minimal = []
                    for size in range(1, len(en) + 1):
                        for S in itertools.combinations(range(len(en)), size):
                            if any(set(M) <= set(S) for M in minimal):
                                continue
                            if tstar(N[list(S)], c[list(S)], fc.v_max, fc.w_max) < 0:
                                minimal.append(S)
                    if not minimal:
                        rows.append(dict(label=LABEL, policy_id=pid, seed=p["seed"], density=K, trial=int(t), robot=int(i),
                                         n_enabled=len(en), t_star=t_all, n_minimal_sets=0, smallest_set_size="",
                                         smallest_set_kinds="", box_needed="", grad_angle_deg="", all_minimal_sets=""))
                        continue
                    S0 = min(minimal, key=len)
                    box_needed = tstar(N[list(S0)], c[list(S0)], 1e3, 1e3) >= 0
                    ang = ""
                    if len(S0) == 2:
                        ang = float(np.degrees(np.arccos(np.clip(N[S0[0]] @ N[S0[1]], -1, 1))))
                    rows.append(dict(label=LABEL, policy_id=pid, seed=p["seed"], density=K, trial=int(t), robot=int(i),
                                     n_enabled=len(en), t_star=t_all, n_minimal_sets=len(minimal), smallest_set_size=len(S0),
                                     smallest_set_kinds="+".join(sorted(kinds[s] for s in S0)), box_needed=int(box_needed),
                                     grad_angle_deg=ang,
                                     all_minimal_sets="|".join("+".join(sorted(kinds[s] for s in M)) for M in minimal)))
    OUT.mkdir(exist_ok=True)
    keys = list(rows[0].keys())
    with open(OUT / "infeasibility_conflicts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["policy_id"], r["density"], r["trial"], r["robot"])):
            w.writerow({k: (f"{v:.6e}" if isinstance(v, float) else v) for k, v in r.items()})
    ts = np.array([r["t_star"] for r in rows])
    conf = [r for r in rows if r["n_minimal_sets"]]
    kinds = {}
    for r in conf:
        kinds[r["smallest_set_kinds"]] = kinds.get(r["smallest_set_kinds"], 0) + 1
    angs = np.array([r["grad_angle_deg"] for r in conf if r["grad_angle_deg"] != ""], dtype=float)
    summ = dict(label=LABEL, robots=len(rows), t_star_negative=int((ts < 0).sum()),
                t_star_below_solver_tol=int((ts < -fc.solver_tol).sum()),
                t_star_quantiles=[float(np.quantile(ts, q)) for q in (0, 0.25, 0.5, 0.75, 1)],
                with_minimal_set=len(conf), smallest_set_size_counts={str(s): sum(r["smallest_set_size"] == s for r in conf) for s in (1, 2, 3, 4, 5)},
                smallest_set_kinds=kinds, box_needed=sum(r["box_needed"] == 1 for r in conf),
                pair_angle_deg_quantiles=[float(np.quantile(angs, q)) for q in (0, 0.25, 0.5, 0.75, 1)] if len(angs) else [],
                pair_conflicts=int(len(angs)))
    (OUT / "infeasibility_conflicts_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
