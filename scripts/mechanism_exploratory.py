#!/usr/bin/env python3
"""EXPLORATORY, POST HOC mechanism analysis of the final test (commits e23f8c3, 41ae3d4).

Reads only the saved per-trial cells and per-step logs; runs no simulation and changes no
confirmatory result. Barrier rows are recomputed from the SAVED pre-step poses with the unchanged
row construction (cbf_study.bmatch.barrier_rows, identical to CBFQPFilter's rows at alpha* = 10).

1. TIMEOUT classification (filter on, B-match), definitions fixed here, post hoc:
   window = the last 50 executed periods (the protocol's stall window);
   'active' period = at least one robot's clipped NOMINAL command violates an enabled barrier row
   (the trigger for a filter intervention / brake activation), obstacle and neighbour rows separately;
   * slow progress      : protocol stalled flag = 0 (centroid still progressing >= 0.05 m over the window)
   * filter-induced stall: stalled and active in >= 50 % of window periods (split: obstacle / neighbour / both)
   * near-goal parking stall: stalled, not filter-induced, final centroid distance <= 0.30 m (Phase 1 definition)
   * other stall        : stalled, not filter-induced, final centroid distance > 0.30 m
   paired with the filter-off outcome on the same layout.
2. Infeasible QPs: state at the attempted period (pose index = executed periods).
3. Closing speed along each robot pair's barrier-point axis, s_i = -e_ij . pdot_i with pdot_i = J(theta_i) u_i
   (applied command), against the equal-split cap c_i = ((alpha/2) h_ij - m_ij) / (2 d_ij), m_ij = delta_ij / 2,
   alpha = 10 (row i reads s_i <= c_i). Formation transit = executed periods with centroid > 0.30 m from the goal.

Writes results/final_test/mechanism/*.csv and results/final_test/MECHANISM_EXPLORATORY.md."""
import csv, gzip, json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbf_study.bmatch import barrier_rows, rows_violated     # noqa: E402
from cbf_study.cbf_qp import barrier_point, margin_robot, rot_jacobian   # noqa: E402
from cbf_study.config import StudyConfig                     # noqa: E402
from cbf_study.evaluate import write_rows                    # noqa: E402

FT = ROOT / "results" / "final_test"
OUT = FT / "mechanism"
WINDOW, ACTIVE_SHARE, NEAR, MOVING = 50, 0.5, 0.30, 0.01
LABEL = "EXPLORATORY, POST HOC"


def load_cells():
    rows = {}
    for p in sorted((FT / "cells").glob("*.csv.gz")):
        for r in csv.DictReader(gzip.open(p, "rt")):
            rows[(r["policy_id"], int(r["density"]), r["condition"], int(r["trial"]))] = r
    return rows


def main():
    cfg = StudyConfig.load()
    fc = cfg.filter_config(10.0)
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())
    pols = [p["policy_id"] for p in sel["policies"]]
    seed_of = {p["policy_id"]: p["seed"] for p in sel["policies"]}
    cells = load_cells()
    OUT.mkdir(parents=True, exist_ok=True)
    to_rows, inf_rows, speed = [], [], defaultdict(lambda: defaultdict(list))
    for pid in pols:
        for K in (1, 2, 3):
            lay = np.load(FT / "layouts" / f"{pid}_K{K}.npz")
            obst_all, goal = lay["obst_xy"], lay["goal"]
            for cond in ("off", "on", "bmatch"):
                z = np.load(FT / "steps" / f"{pid}_K{K}_{cond}.npz")
                T, att, po, co = z["executed_periods"], z["attempted_periods"], z["pose_offsets"], z["cmd_offsets"]
                poses = z["poses"].astype(np.float64)
                trial = np.repeat(np.arange(len(T)), att)                     # trial of every command row
                k_in = np.arange(len(trial)) - co[trial]                      # period index within the trial
                pre = poses[po[trial] + k_in]                                 # pre-step pose of every command row
                obst = obst_all[trial]
                A, b, m = barrier_rows(fc, pre, obst)
                clip = np.nan_to_num(z["clipped"].astype(np.float64))
                v_obs = rows_violated(clip, A[:, :, :K], b[:, :, :K], m[:, :, :K], fc.solver_tol).any(axis=1)
                v_nb = rows_violated(clip, A[:, :, K:], b[:, :, K:], m[:, :, K:], fc.solver_tol).any(axis=1)
                executed = k_in < T[trial]
                # ---- 3. closing speed vs equal-split cap (executed periods in formation transit) ----
                cen_d = np.linalg.norm(pre[..., :2].mean(axis=1) - goal[trial], axis=1)
                tr = executed & (cen_d > NEAR)
                u = np.nan_to_num(z["applied"])
                p = barrier_point(pre, fc.ell)
                pdot = np.einsum("rnij,rnj->rni", rot_jacobian(pre[..., 2], fc.ell), u)
                for i in range(3):
                    for j in range(3):
                        if i == j:
                            continue
                        q = p[:, i] - p[:, j]
                        d = np.linalg.norm(q, axis=1)
                        e = q / d[:, None]
                        s = -(e * pdot[:, i]).sum(axis=1)                    # closing speed of i toward j
                        h = d ** 2 - fc.d_rr ** 2
                        cap = (0.5 * fc.alpha * h - 0.5 * margin_robot(fc, d)) / (2 * d)
                        sel_ = tr & (d <= fc.comm_range) & (s > MOVING)       # i is the trailing (closing) robot, actually moving
                        key = (cond, K)
                        speed[key]["s"].append(s[sel_]); speed[key]["cap"].append(cap[sel_])
                        speed[key]["d"].append(d[sel_])
                # ---- 1. TIMEOUT classification ----
                if cond != "off":
                    for t in np.flatnonzero(z["outcome"] == 4):
                        r = cells[(pid, K, cond, int(t))]
                        lo, hi = co[t] + max(0, T[t] - WINDOW), co[t] + T[t]
                        so, sn = v_obs[lo:hi].mean(), v_nb[lo:hi].mean()
                        sa = (v_obs[lo:hi] | v_nb[lo:hi]).mean()
                        dist = float(r["goal_dist_final"])
                        if r["stalled"] == "0":
                            cat = "slow progress"
                        elif sa >= ACTIVE_SHARE:
                            cat = ("filter-induced stall: " + ("obstacle+neighbour" if so >= ACTIVE_SHARE and sn >= ACTIVE_SHARE
                                   else "obstacle" if so >= sn else "neighbour"))
                        elif dist <= NEAR:
                            cat = "near-goal parking stall"
                        else:
                            cat = "other stall"
                        to_rows.append(dict(label=LABEL, policy_id=pid, seed=seed_of[pid], density=K, condition=cond, trial=int(t),
                                            category=cat, near_goal=int(dist <= NEAR), final_goal_dist=dist,
                                            window_active_share=float(sa), window_obstacle_share=float(so),
                                            window_neighbour_share=float(sn), off_outcome=cells[(pid, K, "off", int(t))]["outcome"]))
                # ---- 2. infeasible QPs ----
                if cond == "on":
                    for t in np.flatnonzero(z["outcome"] == 3):
                        row = co[t] + att[t] - 1                              # the attempted (unexecuted) period
                        assert z["infeasible"][row].any() and k_in[row] == T[t]
                        ps = pre[row]
                        pb = barrier_point(ps[None], fc.ell)[0]
                        for i in np.flatnonzero(z["infeasible"][row]):
                            do = np.linalg.norm(pb[i] - obst_all[t], axis=1) if K else np.array([])
                            dn = np.array([np.linalg.norm(pb[i] - pb[j]) for j in range(3) if j != i])
                            en_o, en_n = int(m[row, i, :K].sum()), int(m[row, i, K:].sum())
                            cfgname = (f"{en_o} obstacle + {en_n} neighbour rows")
                            inf_rows.append(dict(label=LABEL, policy_id=pid, seed=seed_of[pid], density=K, trial=int(t), robot=int(i),
                                                 period=int(T[t]), enabled_obstacle_rows=en_o, enabled_neighbour_rows=en_n,
                                                 configuration=cfgname,
                                                 nominal_violates_obstacle=int(rows_violated(clip[row][None, i:i + 1], A[row][None, i:i + 1, :K], b[row][None, i:i + 1, :K], m[row][None, i:i + 1, :K], fc.solver_tol)[0, 0]),
                                                 nominal_violates_neighbour=int(rows_violated(clip[row][None, i:i + 1], A[row][None, i:i + 1, K:], b[row][None, i:i + 1, K:], m[row][None, i:i + 1, K:], fc.solver_tol)[0, 0]),
                                                 min_obstacle_surface_gap=float((np.linalg.norm(ps[i, :2] - obst_all[t], axis=1) - fc.robot_r - fc.obst_r).min()) if K else np.nan,
                                                 min_neighbour_surface_gap=float(min(np.linalg.norm(ps[i, :2] - ps[j, :2]) - 2 * fc.robot_r for j in range(3) if j != i)),
                                                 min_obstacle_bp_dist=float(do.min()) if K else np.nan, min_neighbour_bp_dist=float(dn.min()),
                                                 centroid_goal_dist=float(np.linalg.norm(ps[:, :2].mean(axis=0) - goal[t])),
                                                 off_outcome=cells[(pid, K, "off", int(t))]["outcome"]))
            print(pid, "done", flush=True)
    write_rows(OUT / "timeout_classes.csv", to_rows)
    write_rows(OUT / "infeasible_qps.csv", inf_rows)
    sp = []
    for (cond, K), v in sorted(speed.items()):
        s, cap, d = (np.concatenate(v[k]) for k in ("s", "cap", "d"))
        sp.append(dict(label=LABEL, condition=cond, density=K, closing_robot_pair_periods=len(s),
                       s_median=float(np.median(s)), s_q90=float(np.quantile(s, 0.9)), cap_median=float(np.median(cap)),
                       share_above_cap=float((s > cap + 1e-6).mean()), share_binding=float((s >= cap - 1e-6).mean()),
                       share_within_10pct_of_cap=float((s >= 0.9 * cap).mean()), ratio_median=float(np.median(s / cap)),
                       d_median=float(np.median(d)),
                       cap_at_median_d=float((0.5 * fc.alpha * (np.median(d) ** 2 - fc.d_rr ** 2) - 0.5 * margin_robot(fc, np.median(d))) / (2 * np.median(d)))))
    write_rows(OUT / "closing_speed_vs_cap.csv", sp)
    write_md(to_rows, inf_rows, sp, pols, fc)


def write_md(to_rows, inf_rows, sp, pols, fc):
    cats = ["near-goal parking stall", "filter-induced stall: neighbour", "filter-induced stall: obstacle",
            "filter-induced stall: obstacle+neighbour", "other stall", "slow progress"]
    L = [f"# Mechanism analysis of the final test: {LABEL}", "",
         f"**{LABEL}.** Computed from the saved final-test logs (commits e23f8c3, 41ae3d4); no simulation was run and "
         "no confirmatory result changes (the verdict in `FINAL_TEST.md` stands). Categories, thresholds and quantities were "
         "chosen after the confirmatory results were seen. Generated by `scripts/mechanism_exploratory.py`; tables in "
         "`mechanism/`.", ""]
    on = [r for r in to_rows if r["condition"] == "on"]
    fi = [r for r in on if r["category"].startswith("filter-induced")]
    ng = [r for r in on if r["category"] == "near-goal parking stall"]
    bm = [r for r in to_rows if r["condition"] == "bmatch"]
    fib = [r for r in bm if r["category"].startswith("filter-induced")]
    oc = Counter(r["off_outcome"] for r in fi)
    trials_inf = {(r["policy_id"], r["density"], r["trial"]): r for r in inf_rows}
    inf_off_coll = sum(1 for r in trials_inf.values() if r["off_outcome"] == "COLLISION")
    obs_inf = sum(1 for r in inf_rows if r["nominal_violates_obstacle"])
    L += ["## Plain-language summary (exploratory, post hoc)", "",
          f"* **Timeouts.** With the filter on, {len(on):,} trials timed out. {len(fi):,} ({100 * len(fi) / len(on):.0f}%) are "
          f"filter-induced stalls: the team sits in front of obstacles while the actor keeps commanding motion the barrier rows forbid "
          f"(obstacle rows involved in {sum(1 for r in fi if 'obstacle' in r['category']):,}). On the same layouts the filter-off "
          f"run collided in {oc['COLLISION']:,}, reached the goal in {oc['GOAL']:,} and timed out in {oc['TIMEOUT']:,}. "
          f"{len(ng):,} ({100 * len(ng) / len(on):.0f}%) are near-goal parking stalls, {sum(1 for r in ng if r['off_outcome'] == 'TIMEOUT'):,} "
          "of them on layouts where the filter-off run stalled the same way: a property of the actor, not of the filter. Very few "
          "timeouts are still moving at the horizon. B-match shows the same pattern with more filter-induced stalls "
          f"({len(fib):,} of {len(bm):,}).",
          f"* **Infeasible QPs** ({len(trials_inf)} trials) occur early (median period 36-60), near the entry of the obstacle "
          "field (centroid about 2.1-2.7 m from the goal), for a robot that has two neighbour rows and one to three obstacle rows "
          f"enabled; the nominal command violates an obstacle row in {obs_inf} of {len(inf_rows)} cases, with the robot about "
          f"0.08 m from an obstacle surface and 0.12 m from a neighbour. On {inf_off_coll} of these layouts the filter-off run "
          "went on to collide.",
          "* **Equal-split cap.** At alpha = 10 the robot-robot cap on closing speed is rarely binding during transit: its median "
          f"(about {min(r['cap_median'] for r in sp if r['condition'] == 'on'):.2f}-{max(r['cap_median'] for r in sp if r['condition'] == 'on'):.2f} m/s) "
          f"exceeds the largest barrier-point speed the velocity box allows ({fc.vp:.2f} m/s), the closing robot moves at about 0.4 "
          "of it (median), and it binds in 1-8% of moving closing periods (filter-off motion exceeds it in 1.5-11.6%). The "
          "formation and GOAL losses at alpha* appear to come mainly from obstacle rows, not from trailing robots being held back "
          "by the equal split.",
          "* These readings are post hoc and descriptive. They do not change the confirmatory verdict, and they say nothing "
          "about other gains, actors or a modified filter.", ""]
    L += ["## 1. Why trials time out (filter on, B-match), paired with filter off (Stage 2 H3)", "",
          f"Window = last {WINDOW} executed periods. A period is 'active' when a robot's clipped nominal command violates an "
          f"enabled barrier row (alpha = 10). Filter-induced stall = protocol stall flag and active in >= {int(100 * ACTIVE_SHARE)}% "
          f"of the window; near-goal parking stall = stalled, not filter-induced, final centroid within {NEAR} m of the goal "
          "(Phase 1 definition); slow progress = not stalled (still moving at the horizon).", ""]
    for cond in ("on", "bmatch"):
        rs = [r for r in to_rows if r["condition"] == cond]
        L += [f"### {'Filter on' if cond == 'on' else 'B-match'}: TIMEOUT trials by category and density", "",
              "| category | " + " | ".join(f"K = {K}" for K in (1, 2, 3)) + " | total |", "|---|---|---|---|---|"]
        for c in cats:
            n = [sum(1 for r in rs if r["category"] == c and r["density"] == K) for K in (1, 2, 3)]
            L.append(f"| {c} | " + " | ".join(map(str, n)) + f" | {sum(n)} |")
        n = [sum(1 for r in rs if r["density"] == K) for K in (1, 2, 3)]
        L.append("| **all TIMEOUT** | " + " | ".join(map(str, n)) + f" | {sum(n)} |")
        L += ["", "Paired with the filter-off outcome on the same layout (all densities):", "",
              "| category | off GOAL | off COLLISION | off TIMEOUT |", "|---|---|---|---|"]
        for c in cats:
            cc = Counter(r["off_outcome"] for r in rs if r["category"] == c)
            L.append(f"| {c} | {cc['GOAL']} | {cc['COLLISION']} | {cc['TIMEOUT']} |")
        L += ["", "Per seed and per actor (counts over K = 1, 2, 3):", "",
              "| scope | " + " | ".join(cats) + " |", "|---|" + "---|" * len(cats)]
        for s in (0, 1, 2):
            L.append(f"| seed{s} | " + " | ".join(str(sum(1 for r in rs if r["seed"] == s and r["category"] == c)) for c in cats) + " |")
        for p in pols:
            L.append(f"| {p} | " + " | ".join(str(sum(1 for r in rs if r["policy_id"] == p and r["category"] == c)) for c in cats) + " |")
        L.append("")
    L += ["## 2. Infeasible QPs (filter on)", "",
          "State at the attempted period that infeasibility prevented. Rows enabled for the infeasible robot; gaps are surface "
          "clearances (centre distance minus radii).", "",
          "| K | infeasible trials (robots) | configuration (enabled rows): count | nominal violates obstacle / neighbour rows | period median [min, max] | min neighbour gap median [m] | min obstacle gap median [m] | centroid-goal distance median [m] | filter-off outcome GOAL / COLL / TIMEOUT |",
          "|---|---|---|---|---|---|---|---|---|"]
    for K in (1, 2, 3):
        rs = [r for r in inf_rows if r["density"] == K]
        if not rs:
            continue
        cc = Counter(r["configuration"] for r in rs)
        trials = {(r["policy_id"], r["trial"]): r for r in rs}
        oc = Counter(r["off_outcome"] for r in trials.values())
        per = [r["period"] for r in trials.values()]
        med = lambda k: float(np.nanmedian([r[k] for r in rs]))
        L.append(f"| {K} | {len(trials)} ({len(rs)}) | " + "; ".join(f"{k}: {v}" for k, v in cc.most_common()) + " | "
                 f"{sum(r['nominal_violates_obstacle'] for r in rs)} / {sum(r['nominal_violates_neighbour'] for r in rs)} | "
                 f"{np.median(per):.0f} [{min(per)}, {max(per)}] | {med('min_neighbour_surface_gap'):.3f} | {med('min_obstacle_surface_gap'):.3f} | "
                 f"{med('centroid_goal_dist'):.2f} | {oc['GOAL']} / {oc['COLLISION']} / {oc['TIMEOUT']} |")
    L += ["", "## 3. Closing speed along the pair axis vs the equal-split cap (alpha = 10)", "",
          "For every robot pair within communication range, in executed periods of formation transit (centroid > 0.30 m from "
          "the goal), the closing robot's barrier-point speed (only robot-pair-periods with s > 0.01 m/s) toward its neighbour s = -e.pdot (applied command) against the "
          "equal-split cap c = ((alpha/2) h - m) / (2 d), m = delta/2. The filter enforces s <= c for each robot (up to the "
          "solver tolerance); filter off and B-match are not bound by it.", "",
          "| condition | K | closing robot-pair-periods (s > 0.01 m/s) | s median / q90 [m/s] | cap median [m/s] | s / cap median | share above cap | share binding (s >= cap - 1e-6) | share >= 90% of cap | pair distance median [m] |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sp:
        L.append(f"| {r['condition']} | {r['density']} | {r['closing_robot_pair_periods']:,} | {r['s_median']:.4f} / {r['s_q90']:.4f} | "
                 f"{r['cap_median']:.4f} | {r['ratio_median']:.3f} | {100 * r['share_above_cap']:.2f}% | {100 * r['share_binding']:.2f}% | "
                 f"{100 * r['share_within_10pct_of_cap']:.2f}% | {r['d_median']:.3f} |")
    (FT / "MECHANISM_EXPLORATORY.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
