#!/usr/bin/env python3
"""Paper-revision analyses on SAVED data only (no simulation, no QP or filter call, no new layouts).

(a) Time to goal - "pre-declared episode-length endpoint; exploratory, post hoc joint-success analysis".
    Pairs (same actor, density, layout) where BOTH the filter-off arm and the test arm (filter on, or
    B-match) reached GOAL; difference = test-arm executed periods minus filter-off executed periods
    (positive = slower). Per actor: joint-success count and mean difference. Equal-actor-weighted mean
    = mean of the actor means (unavailable if any actor in the scope has no joint-success pair).
    Pooled median, Wilcoxon signed-rank p and Hodges-Lehmann (with its signed-rank CI) are
    PAIR-weighted. Conditioning on joint success does not estimate the overall completion-time cost.
(b) Per-seed locked endpoints (pre-declared secondary, descriptive): from results/final_test/endpoints.csv.
(c) Validation vs final test at alpha = 10 (exploratory, post hoc).
(d) Stress cells K = 4, 5, 8 from the validation sweep (exploratory, post hoc, descriptive).
(e) Filter-off baseline per actor at K = 1, 2, 3, 4, 5, 8 (validation, results/phase1).
(f) Example trajectory selection (exploratory, post hoc): group-A pair whose filter-off minimum sampled
    obstacle clearance is closest to the UNROUNDED group-A median; ties by actor index, density, trial.
Writes results/paper_revision/*.csv, example_selection.json and paper/figures/F6_example_trajectory.pdf."""
import csv, gzip, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbf_study.config import StudyConfig      # noqa: E402
from cbf_study.evaluate import write_rows     # noqa: E402
from cbf_study.stats import hodges_lehmann, wilcoxon_test   # noqa: E402

FT = ROOT / "results" / "final_test"
VS = ROOT / "results" / "validation_sweep"
OUT = ROOT / "results" / "paper_revision"
SEL = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())
POLS = [p["policy_id"] for p in SEL["policies"]]
SEED = {p["policy_id"]: p["seed"] for p in SEL["policies"]}
EXP = "exploratory, post hoc"
TTG = "pre-declared episode-length endpoint; exploratory, post hoc joint-success analysis"


def load_cells():
    c = {}
    for p in sorted((FT / "cells").glob("*.csv.gz")):
        for r in csv.DictReader(gzip.open(p, "rt")):
            c[(r["policy_id"], int(r["density"]), r["condition"], int(r["trial"]))] = (r["outcome"], int(r["executed_periods"]))
    return c


def time_to_goal(cells):
    rows = []
    for test in ("on", "bmatch"):
        per = {}
        for pid in POLS:
            for K in (1, 2, 3):
                d = [cells[(pid, K, test, t)][1] - cells[(pid, K, "off", t)][1] for t in range(1000)
                     if cells[(pid, K, "off", t)][0] == "GOAL" and cells[(pid, K, test, t)][0] == "GOAL"]
                per[(pid, K)] = np.array(d, float)
        scopes = [("density", str(K), [(p, K) for p in POLS]) for K in (1, 2, 3)] \
            + [("seed x density", f"seed{s} K{K}", [(p, K) for p in POLS if SEED[p] == s]) for s in (0, 1, 2) for K in (1, 2, 3)] \
            + [("actor x density", f"{p} K{K}", [(p, K)]) for p in POLS for K in (1, 2, 3)]
        for sc, sid, keys in scopes:
            ds = [per[k] for k in keys]
            allv = np.concatenate(ds) if ds else np.array([])
            missing = [f"{k[0]} K{k[1]}" for k, v in zip(keys, ds) if len(v) == 0]
            eq = float(np.mean([v.mean() for v in ds])) if not missing else float("nan")
            r = dict(label=TTG, test_arm=test, scope=sc, scope_id=sid, joint_success_pairs=len(allv),
                     joint_success_per_actor=";".join(f"{k[0]}:{len(v)}" for k, v in zip(keys, ds)),
                     equal_actor_mean_diff=eq, equal_actor_unavailable_for=";".join(missing),
                     pooled_mean_diff_pairweighted=float(allv.mean()) if len(allv) else float("nan"),
                     pooled_median_diff_pairweighted=float(np.median(allv)) if len(allv) else float("nan"),
                     share_slower=float((allv > 0).mean()) if len(allv) else float("nan"),
                     share_equal=float((allv == 0).mean()) if len(allv) else float("nan"))
            if len(allv) >= 2:
                w, hl = wilcoxon_test(allv), hodges_lehmann(allv)
                r.update(wilcoxon_p_pairweighted=w["p_value"], hl_pairweighted=hl[0], hl_lo=hl[1], hl_hi=hl[2])
            rows.append(r)
    write_rows(OUT / "time_to_goal.csv", rows)
    return rows


def per_seed():
    E = list(csv.DictReader(open(FT / "endpoints.csv")))
    out = []
    for r in E:
        if r["scope"] == "seed" and r["reference"] == "off" and r["test"] == "on":
            n = int(r["n_pairs"])
            out.append(dict(label="pre-declared secondary, descriptive", seed=r["scope_id"], density=r["density"], n_pairs=n,
                            fail_off=(int(r["ref_COLLISION"]) + int(r["ref_INFEASIBLE"])) / n,
                            fail_on=(int(r["test_COLLISION"]) + int(r["test_INFEASIBLE"])) / n,
                            fail_diff=float(r["a_diff"]), fail_lo=float(r["a_ci95_lo"]), fail_hi=float(r["a_ci95_hi"]),
                            goal_loss=float(r["b_loss"]), goal_lo=float(r["b_ci95_lo"]), goal_hi=float(r["b_ci95_hi"]),
                            form_inc=float(r["c_estimate"]), form_lo=float(r["c_ci95_lo"]), form_hi=float(r["c_ci95_hi"])))
    write_rows(OUT / "per_seed_endpoints.csv", out)
    return out


def pooled(E, K, ref, test, alpha=None):
    return next(e for e in E if e["scope"] == "pooled" and e["density"] == str(K) and e["reference"] == ref and e["test"] == test
                and (alpha is None or float(e["alpha"]) == alpha))


def val_vs_test():
    V = list(csv.DictReader(open(VS / "endpoints.csv")))
    F = list(csv.DictReader(open(FT / "endpoints.csv")))
    out = []
    for K in (1, 2, 3):
        for src, E, n in (("validation (development, 200 layouts x 9 actors)", V, None), ("final test (reserved, 1,000 x 9)", F, None)):
            r = pooled(E, K, "off", "on", 10.0)
            n = int(r["n_pairs"])
            out.append(dict(label=EXP, density=K, source=src, n_pairs=n,
                            fail_off=(int(r["ref_COLLISION"]) + int(r["ref_INFEASIBLE"])) / n,
                            fail_on=(int(r["test_COLLISION"]) + int(r["test_INFEASIBLE"])) / n,
                            goal_off=int(r["ref_GOAL"]) / n, goal_on=int(r["test_GOAL"]) / n,
                            goal_loss=float(r["b_loss"]), goal_lo=float(r["b_ci95_lo"]), goal_hi=float(r["b_ci95_hi"]),
                            form_inc=float(r["c_estimate"]), form_lo=float(r["c_ci95_lo"]), form_hi=float(r["c_ci95_hi"])))
    write_rows(OUT / "validation_vs_test.csv", out)
    return out


def stress():
    V = list(csv.DictReader(open(VS / "endpoints.csv")))
    out = []
    for K in (4, 5, 8):
        ron, rbm = pooled(V, K, "off", "on", 10.0), pooled(V, K, "off", "bmatch", 10.0)
        n = int(ron["n_pairs"])
        for arm, r, side in (("filter off", ron, "ref"), ("filter on (alpha = 10)", ron, "test"), ("B-match (alpha = 10)", rbm, "test")):
            out.append(dict(label=EXP + "; validation layouts; not part of the final test", density=K, arm=arm, n=n,
                            **{o: int(r[f"{side}_{o}"]) / n for o in ("GOAL", "COLLISION", "INFEASIBLE", "TIMEOUT")}))
    write_rows(OUT / "stress_cells.csv", out)
    return out


def baseline():
    rows = list(csv.DictReader(open(ROOT / "results/phase1/tables/filter_off_by_policy_density.csv")))
    out = [dict(label="filter-off validation baseline (Phase 1, results/phase1)", policy_id=r["policy_id"], density=int(r["density"]),
                n=int(r["n"]), GOAL=int(r["GOAL"]) / int(r["n"]), COLLISION=(int(r["COLL_OBST"]) + int(r["COLL_ROBOT"])) / int(r["n"]),
                TIMEOUT=int(r["TIMEOUT"]) / int(r["n"])) for r in rows]
    write_rows(OUT / "filter_off_baseline.csv", out)
    return out


def example(cfg):
    import matplotlib
    matplotlib.use("pdf")
    import matplotlib.pyplot as plt
    rows = [r for r in csv.DictReader(open(FT / "mechanism/clearance_trajectories.csv")) if r["in_A"] == "1"]
    g = np.array([float(r["min_gap_obst"]) for r in rows])
    med = float(np.median(g))
    key = lambda r: (abs(float(r["min_gap_obst"]) - med), POLS.index(r["policy_id"]), int(r["density"]), int(r["trial"]))
    best = min(rows, key=key)
    ties = [r for r in rows if abs(float(r["min_gap_obst"]) - med) == key(best)[0]]
    pid, K, t = best["policy_id"], int(best["density"]), int(best["trial"])
    pc, fc = cfg.protocol_config(), cfg.filter_config(10.0)
    R, Ro, ell = pc.robot_r, pc.obst_r, fc.ell
    obst = np.load(FT / "layouts" / f"{pid}_K{K}.npz")["obst_xy"][t]
    goal = np.load(FT / "layouts" / f"{pid}_K{K}.npz")["goal"][t]

    def traj(cond):
        z = np.load(FT / "steps" / f"{pid}_K{K}_{cond}.npz")
        T, po = int(z["executed_periods"][t]), z["pose_offsets"]
        return z["poses"][po[t]:po[t] + T + 1].astype(np.float64), T, int(z["outcome"][t])
    Soff, Toff, ooff = traj("off")
    Son, Ton, oon = traj("on")
    gap_off = np.linalg.norm(Soff[:, :, None, :2] - obst[None, None], axis=3) - R - Ro       # [T+1, N, K]
    k_ca = int(np.argmin(gap_off.min(axis=(1, 2))))
    cen = Son[..., :2].mean(axis=1)
    d = np.linalg.norm(cen - cen[Ton], axis=1)
    far = np.flatnonzero(d > 0.05)
    k_st = int(far[-1] + 1) if len(far) else 0                                                 # existing stall-onset convention
    pb = Son[k_st, :, :2] + ell * np.stack([np.cos(Son[k_st, :, 2]), np.sin(Son[k_st, :, 2])], 1)
    h = np.linalg.norm(pb[:, None] - obst[None], axis=2) ** 2 - fc.d_obst ** 2                # [N, K]
    hmin = h.min(axis=1)
    i_sel = int(np.flatnonzero(hmin == hmin.min())[0])                                         # ties -> lowest robot index
    j_obs = int(np.argmin(h[i_sel]))
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 3.0), sharex=True, sharey=True)
    for ax, S, k, title in ((axs[0], Soff, k_ca, f"filter off (GOAL), closest approach t = {k_ca}"),
                            (axs[1], Son, k_st, f"filter on (TIMEOUT), stall onset t = {k_st}")):
        for c in obst:
            ax.add_patch(plt.Circle(c, Ro, color="0.6"))
        for i, col in enumerate(("#4c72b0", "#dd8452", "#55a868")):
            ax.plot(S[:, i, 0], S[:, i, 1], lw=0.8, color=col)
            ax.add_patch(plt.Circle(S[k, i, :2], R, fill=False, color=col, lw=1.2))
        ax.plot(*goal, "k*", ms=6)
        ax.set_aspect("equal"); ax.set_title(title, fontsize=7)
    axs[1].add_patch(plt.Circle(pb[i_sel], R + ell, fill=False, ls="--", color="k", lw=0.9))
    axs[1].plot(*pb[i_sel], "k.", ms=3)
    axs[0].set_xlim(-2.0, 1.9); axs[0].set_ylim(-1.2, 1.2)
    for ax in axs:
        ax.set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    fig.tight_layout(); fig.savefig(ROOT / "paper/figures/F6_example_trajectory.pdf"); plt.close(fig)
    sel = dict(label=EXP, rule="group-A pair with filter-off minimum sampled obstacle clearance closest to the unrounded group-A "
                               "median; ties by actor index (checkpoint_selection order), density, trial index",
               group_A_n=len(rows), group_A_median_m=med, n_tied_at_min_distance=len(ties), policy_id=pid, density=K, trial=t,
               selected_min_gap_obst_m=float(best["min_gap_obst"]), off_executed_periods=Toff, on_executed_periods=Ton,
               off_outcome_code=ooff, on_outcome_code=oon, off_closest_approach_state=k_ca,
               off_min_clearance_at_that_state_m=float(gap_off[k_ca].min()),
               on_stall_onset_state=k_st, stall_onset_convention="first state from which the filter-on centroid stays within 0.05 m of its final position",
               cover_disc_robot=i_sel, cover_disc_robot_rule="robot with the smallest obstacle cover barrier at stall onset; ties by robot index",
               cover_disc_robot_h_m2=float(hmin[i_sel]), cover_disc_obstacle=j_obs,
               cover_disc_robot_body_clearance_m=float(np.linalg.norm(Son[k_st, i_sel, :2] - obst[j_obs]) - R - Ro),
               note="geometric selection only; it does not show that this robot or constraint caused the stall, and a centroid stall does not mean every robot stopped")
    (OUT / "example_selection.json").write_text(json.dumps(sel, indent=1))
    return sel


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = StudyConfig.load()
    cells = load_cells()
    ttg = time_to_goal(cells)
    per_seed(); val_vs_test(); stress(); baseline()
    sel = example(cfg)
    print(json.dumps(sel, indent=1))
    for r in ttg:
        if r["scope"] == "density":
            print(r["test_arm"], r["scope_id"], r["joint_success_pairs"], round(r["equal_actor_mean_diff"], 2), r["pooled_median_diff_pairweighted"],
                  f"{r.get('wilcoxon_p_pairweighted', float('nan')):.2e}", r.get("hl_pairweighted"))


if __name__ == "__main__":
    main()
