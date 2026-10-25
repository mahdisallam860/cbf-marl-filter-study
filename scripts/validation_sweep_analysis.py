#!/usr/bin/env python3
"""Analysis of the locked development sweep (results/validation_sweep/cells/).

1. Applies the locked gain-selection rule (D4) to K = 1, 2, 3 only and writes alpha_star.json.
2. Reports every gain on every locked endpoint (D2 definitions, applied to development data, so
   descriptive: no confirmatory claim), pooled over the nine actors, per seed and per actor, per
   density; comparisons: filter on vs off, B-match vs off, filter on vs B-match (same gain).
3. Compares alpha = 0.5 and 1 with the recorded analytic prediction (zero-command separations
   0.668 m and 0.501 m vs nominal spacing 0.520 m).

Writes results/validation_sweep/{alpha_star.json, selection.csv, endpoints.csv, VALIDATION_SWEEP.md}."""
import csv, glob, gzip, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbf_study import locked_stats as LS               # noqa: E402
from cbf_study.evaluate import write_rows               # noqa: E402
from cbf_study.stats import hodges_lehmann, newcombe_paired_ci, paired_table, wilcoxon_test   # noqa: E402

OUT = ROOT / "results" / "validation_sweep"
ALPHAS = (0.5, 1.0, 2.0, 5.0, 10.0)
PRIMARY, OTHER = (1, 2, 3), (4, 5, 8)
INT = ("trial", "density", "seed", "goal", "fail", "executed_periods", "stalled", "reached_goal_region", "n_interventions",
       "n_brake_activations", "n_infeasible_robot_steps", "L_vs_off", "L_on_bmatch")
STR = ("policy_id", "phase", "role", "condition", "alpha", "layout_hash", "outcome", "collision_kind", "excl_vs_off", "excl_on_bmatch")


def load():
    data = {}
    for p in sorted(glob.glob(str(OUT / "cells" / "*.csv.gz"))):
        with gzip.open(p, "rt") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            key = (r["policy_id"], int(r["density"]), r["condition"], float(r["alpha"]) if r["alpha"] else None)
            data.setdefault(key, []).append(r)
    arr = {}
    for key, rows in data.items():
        rows.sort(key=lambda r: int(r["trial"]))
        a = {}
        for k in rows[0]:
            if k in STR:
                a[k] = np.array([r[k] for r in rows], dtype=object)
            elif k in INT:
                a[k] = np.array([int(r[k]) for r in rows])
            else:
                a[k] = np.array([float(r[k]) if r[k] not in ("", "nan") else np.nan for r in rows])
        arr[key] = a
    return arr


def cat(arr, pols, K, cond, alpha, field):
    return np.concatenate([arr[(p, K, cond, alpha)][field] for p in pols])


def pol_labels(arr, pols, K, cond, alpha):
    return np.concatenate([np.full(len(arr[(p, K, cond, alpha)]["trial"]), p, dtype=object) for p in pols])


def check_pairs(arr, pols, K, cond_a, al_a, cond_b, al_b):
    for p in pols:
        A, B_ = arr[(p, K, cond_a, al_a)], arr[(p, K, cond_b, al_b)]
        assert (A["trial"] == B_["trial"]).all() and (A["layout_hash"] == B_["layout_hash"]).all(), "pairs not identical"


def compare(arr, pols, K, ref, test, scope, scope_id, alpha):
    """test vs ref on identical layouts. ref/test = (condition, alpha). d = test - ref."""
    check_pairs(arr, pols, K, *ref, *test)
    g = lambda arm, f: cat(arr, pols, K, *arm, f)
    lab = pol_labels(arr, pols, K, *test)
    n = len(lab)
    row = dict(scope=scope, scope_id=scope_id, density=K, alpha=alpha, reference=ref[0], test=test[0], n_pairs=n)
    for arm, nm in ((ref, "ref"), (test, "test")):
        oc = g(arm, "outcome")
        for o in ("GOAL", "COLLISION", "INFEASIBLE", "TIMEOUT"):
            row[f"{nm}_{o}"] = int((oc == o).sum())
        ck = g(arm, "collision_kind")
        row[f"{nm}_coll_obstacle"], row[f"{nm}_coll_robot"] = int((ck == "obstacle").sum()), int((ck == "robot").sum())
        row[f"{nm}_stalled"] = int(g(arm, "stalled").sum())
        row[f"{nm}_reached_goal_region"] = int(g(arm, "reached_goal_region").sum())
        T = g(arm, "executed_periods")
        row[f"{nm}_episode_len_mean"] = float(T.mean())
        iv = g(arm, "n_interventions") + g(arm, "n_brake_activations")
        row[f"{nm}_interv_or_brake_per_trial"] = float(iv.mean())
        row[f"{nm}_interv_or_brake_frac"] = float(iv.sum() / max(1, (T * 3).sum()))
        row[f"{nm}_h_obst_min_prestep"] = float(np.min(g(arm, "h_obst_min_prestep")))
        row[f"{nm}_h_rr_min_prestep"] = float(np.min(g(arm, "h_rr_min_prestep")))
        row[f"{nm}_h_obst_min_sampled"] = float(np.min(g(arm, "h_obst_min_sampled")))
        row[f"{nm}_h_rr_min_sampled"] = float(np.min(g(arm, "h_rr_min_sampled")))
    # (a) failure, (b) GOAL loss: locked definitions (descriptive on development data)
    a = LS.failure_reduction(g(ref, "fail").astype(bool), g(test, "fail").astype(bool), lab)
    b = LS.goal_noninferiority(g(ref, "goal").astype(bool), g(test, "goal").astype(bool), lab)
    row.update(a_diff=a["diff_on_minus_off"], a_p_one_sided=a["p_one_sided"], a_upper_9833=a["upper_one_sided"],
               a_ci95_lo=a["ci95"][0], a_ci95_hi=a["ci95"][1], a_meets=int(a["passed"]),
               b_loss=b["loss"], b_upper_9833=b["upper_one_sided"], b_ci95_lo=b["ci95_loss"][0], b_ci95_hi=b["ci95_loss"][1],
               b_meets=int(b["passed"]))
    # (c) common-prefix formation difference (test - ref)
    if ref[0] == "off":
        d, L, why = g(test, "d_vs_off"), g(test, "L_vs_off"), g(test, "excl_vs_off")
    else:                                       # ref = B-match, test = filter on: stored on the B-match rows
        d, L, why = g(ref, "d_on_minus_bmatch"), g(ref, "L_on_bmatch"), g(ref, "excl_on_bmatch")
    elig = why == ""
    c = LS.formation_noninferiority(d, elig, lab, list(pols))
    row.update(c_status=c["status"], c_estimate=c["estimate"], c_upper_9833=c["upper_one_sided"],
               c_ci95_lo=c.get("ci95", (np.nan, np.nan))[0], c_ci95_hi=c.get("ci95", (np.nan, np.nan))[1],
               c_normal_upper=c.get("normal_upper_crosscheck", np.nan), c_meets=int(c["passed"]),
               c_eligible=int(elig.sum()), c_excluded=int((~elig).sum()),
               c_excl_reasons=";".join(f"{r}:{int((why == r).sum())}" for r in sorted(set(why[~elig]))),
               L_min=int(L[elig].min()) if elig.any() else -1, L_q25=float(np.quantile(L[elig], 0.25)) if elig.any() else np.nan,
               L_median=float(np.median(L[elig])) if elig.any() else np.nan, L_q75=float(np.quantile(L[elig], 0.75)) if elig.any() else np.nan,
               L_max=int(L[elig].max()) if elig.any() else -1)
    row["usefulness_dev"] = LS.usefulness(a, b, c) if scope == "pooled" else ""
    # secondaries
    dd = g(test, "goal_dist_final") - g(ref, "goal_dist_final")
    w, hl = wilcoxon_test(dd), hodges_lehmann(dd)
    row.update(final_dist_wilcoxon_p=w["p_value"], final_dist_hl=hl[0], final_dist_hl_lo=hl[1], final_dist_hl_hi=hl[2])
    if elig.sum() >= 2:
        w, hl = wilcoxon_test(d[elig]), hodges_lehmann(d[elig])
        row.update(prefix_wilcoxon_p=w["p_value"], prefix_hl=hl[0], prefix_hl_lo=hl[1], prefix_hl_hi=hl[2])
    both = (g(ref, "goal") == 1) & (g(test, "goal") == 1)
    ft = g(test, "full_traj_form_mean")[both] - g(ref, "full_traj_form_mean")[both]
    row.update(both_goal_pairs=int(both.sum()), both_goal_ref_form=float(g(ref, "full_traj_form_mean")[both].mean()) if both.any() else np.nan,
               both_goal_test_form=float(g(test, "full_traj_form_mean")[both].mean()) if both.any() else np.nan,
               both_goal_diff_mean=float(ft.mean()) if both.any() else np.nan,
               both_goal_pairs_per_policy=";".join(f"{p}:{int(both[lab == p].sum())}" for p in pols))
    rr = g(test, "reached_goal_region").astype(bool), g(ref, "reached_goal_region").astype(bool)
    n11, n10, n01, n00 = paired_table(rr[1], rr[0])
    e, lo, hi = newcombe_paired_ci(n11, n10, n01, n00)
    row.update(rgr_diff=e, rgr_ci95_lo=lo, rgr_ci95_hi=hi)
    return row


def main():
    arr = load()
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())
    pols = [p["policy_id"] for p in sel["policies"]]
    seeds = {s: [p["policy_id"] for p in sel["policies"] if p["seed"] == s] for s in (0, 1, 2)}
    have = {(p, K) for (p, K, c, a) in arr}
    assert all((p, K) in have for p in pols for K in PRIMARY + OTHER), "sweep incomplete"

    # ---- 1. locked selection rule on K = 1, 2, 3 only ---------------------------------------------
    selrows = []
    for a in ALPHAS:
        oc = np.concatenate([cat(arr, pols, K, "on", a, "outcome") for K in PRIMARY])
        selrows.append(dict(alpha=a, n=len(oc), GOAL=int((oc == "GOAL").sum()), INFEASIBLE=int((oc == "INFEASIBLE").sum()),
                            COLLISION=int((oc == "COLLISION").sum()), TIMEOUT=int((oc == "TIMEOUT").sum())))
    best = max(selrows, key=lambda r: (r["GOAL"], -r["INFEASIBLE"], r["alpha"]))
    alpha_star = best["alpha"]
    write_rows(OUT / "selection.csv", selrows)
    stress_info = []
    for a in ALPHAS:
        oc = np.concatenate([cat(arr, pols, K, "on", a, "outcome") for K in OTHER])
        stress_info.append(dict(alpha=a, GOAL_K458=int((oc == "GOAL").sum())))
    (OUT / "alpha_star.json").write_text(json.dumps(dict(
        alpha_star=alpha_star, rule="max pooled filter-on GOAL count over 9 actors and K = 1, 2, 3; ties: fewer INFEASIBLE, then larger alpha",
        selection_table=selrows, densities_used=list(PRIMARY), densities_not_used=list(OTHER),
        protocol="docs/PREFILTER_VALIDATION_PROTOCOL_LOCKED_2026-10-25.md", protocol_tag="prefilter-protocol-locked-2026-10-25",
        protocol_commit="9baf6d773b7cf3e6548e02a0c1d9ebc9d21cf893", data="existing validation layouts (development data)",
        note="selected on development data; the final test (reserved block) has not been generated"), indent=1))

    # ---- 2. every gain on every endpoint ----------------------------------------------------------
    ends = []
    for K in PRIMARY + OTHER:
        for a in ALPHAS:
            for ref, test in ((("off", None), ("on", a)), (("off", None), ("bmatch", a)), (("bmatch", a), ("on", a))):
                ends.append(compare(arr, pols, K, ref, test, "pooled", "all9", a))
                for s, ps in seeds.items():
                    ends.append(compare(arr, ps, K, ref, test, "seed", f"seed{s}", a))
                for p in pols:
                    ends.append(compare(arr, [p], K, ref, test, "policy", p, a))
        print(f"K{K} done", flush=True)
    write_rows(OUT / "endpoints.csv", ends)

    # ---- 3. analytic prediction check (alpha = 0.5 and 1, all gains for context) ------------------
    pred = json.loads((ROOT / "results/prefilter_lock/analytic_predictions.json").read_text())
    an = []
    for K in PRIMARY:
        init = cat(arr, pols, K, "off", None, "bp_sep_initial")
        for a in ALPHAS:
            thr = pred["gains"][str(a)]["robot_robot_zero_command_separation"]
            g = lambda f: cat(arr, pols, K, "on", a, f)
            T = g("executed_periods")
            an.append(dict(density=K, alpha=a, zero_cmd_sep_threshold=thr,
                           initial_min_bp_sep_median=float(np.median(init)), share_initial_below_threshold=float((init < thr).mean()),
                           rr_nominal_violation_frac_mean=float(np.nanmean(g("rr_nominal_violation_frac"))),
                           infeasible=int((g("outcome") == "INFEASIBLE").sum()), infeasible_at_step0=int(((g("outcome") == "INFEASIBLE") & (T == 0)).sum()),
                           goal=int(g("goal").sum()), timeout=int((g("outcome") == "TIMEOUT").sum()),
                           bp_sep_min_median_on=float(np.median(g("bp_sep_min"))),
                           bp_sep_min_median_off=float(np.median(cat(arr, pols, K, "off", None, "bp_sep_min"))),
                           interventions_frac=float(g("n_interventions").sum() / max(1, (T * 3).sum()))))
    write_rows(OUT / "analytic_check.csv", an)
    write_md(selrows, alpha_star, stress_info, ends, an, pred)
    print(json.dumps(dict(alpha_star=alpha_star, selection=selrows), indent=1))


def write_md(selrows, alpha_star, stress_info, ends, an, pred):
    f = lambda x, p=3: "-" if x is None or (isinstance(x, float) and x != x) else f"{x:.{p}f}"
    pct = lambda k, n: f"{100 * k / n:.1f}"
    L = ["# Locked development sweep on validation layouts (filter off / on / B-match)", "",
         "Protocol: `docs/PREFILTER_VALIDATION_PROTOCOL_LOCKED_2026-10-25.md` (tag `prefilter-protocol-locked-2026-10-25`, "
         "commit 9baf6d7; author-locked, not advisor-approved). Existing validation layouts (seed 2,000,000 + K, 200 per "
         "density), nine fixed actors, identical layouts across conditions. **Development data: every number below is "
         "descriptive; the D2 criteria are shown as 'meets on development data', not as confirmatory results.** Generated "
         "by `scripts/validation_sweep_analysis.py`; full tables in `endpoints.csv` (pooled, per seed, per actor).", "",
         "## Gain selection (locked rule, K = 1, 2, 3 only)", "",
         "| alpha | trials | filter-on GOAL | INFEASIBLE | COLLISION | TIMEOUT |", "|---|---|---|---|---|---|"]
    for r in selrows:
        L.append(f"| {r['alpha']:g} | {r['n']} | {r['GOAL']} ({pct(r['GOAL'], r['n'])}%) | {r['INFEASIBLE']} | {r['COLLISION']} | {r['TIMEOUT']} |")
    L += ["", f"**alpha\\* = {alpha_star:g} s^-1** (`alpha_star.json`). K = 4, 5, 8 were not used; for the record, their "
          "pooled filter-on GOAL counts per gain: " + ", ".join(f"{s['alpha']:g}: {s['GOAL_K458']}" for s in stress_info) + ".", ""]
    for title, Ks in (("Primary densities", PRIMARY), ("Diagnostic (K = 4) and stress (K = 5, 8): no claims", OTHER)):
        L += [f"## {title}: pooled over the nine actors", "",
              "Rates in % of 1,800 pairs. (a) failure = COLLISION or INFEASIBLE, diff = test - reference with Newcombe 95% CI; "
              "(b) GOAL loss = reference - test, 98.33% upper bound; (c) common-prefix formation increase (test - reference, "
              "equal actor weights) with 98.33% bootstrap bound, eligible pairs; RGR = reached-goal-region.", "",
              "| K | alpha | comparison | GOAL ref / test | fail ref / test | infeasible test | (a) diff [95% CI] | (b) loss (upper) | (c) increase m (upper) [eligible] | RGR ref / test | meets a/b/c |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in [e for e in ends if e["scope"] == "pooled" and e["density"] in Ks]:
            n = r["n_pairs"]
            L.append(f"| {r['density']} | {r['alpha']:g} | {r['test']} vs {r['reference']} | {pct(r['ref_GOAL'], n)} / {pct(r['test_GOAL'], n)} | "
                     f"{pct(r['ref_COLLISION'] + r['ref_INFEASIBLE'], n)} / {pct(r['test_COLLISION'] + r['test_INFEASIBLE'], n)} | "
                     f"{pct(r['test_INFEASIBLE'], n)} | {100 * r['a_diff']:.1f} [{100 * r['a_ci95_lo']:.1f}, {100 * r['a_ci95_hi']:.1f}] | "
                     f"{100 * r['b_loss']:.1f} ({100 * r['b_upper_9833']:.1f}) | {f(r['c_estimate'], 4)} ({f(r['c_upper_9833'], 4)}) [{r['c_eligible']}] | "
                     f"{pct(r['ref_reached_goal_region'], n)} / {pct(r['test_reached_goal_region'], n)} | {r['a_meets']}/{r['b_meets']}/{r['c_meets']} |")
        L.append("")
    L += ["## Per seed and per actor at alpha*, filter on vs off (K = 1, 2, 3; descriptive)", "",
          "| scope | K | GOAL off / on % | fail off / on % | (b) loss % [95% CI] | (c) increase m [95% CI] |", "|---|---|---|---|---|---|"]
    for r in [e for e in ends if e["scope"] in ("seed", "policy") and e["alpha"] == alpha_star and e["test"] == "on"
              and e["reference"] == "off" and e["density"] in PRIMARY]:
        n = r["n_pairs"]
        L.append(f"| {r['scope_id']} | {r['density']} | {pct(r['ref_GOAL'], n)} / {pct(r['test_GOAL'], n)} | "
                 f"{pct(r['ref_COLLISION'] + r['ref_INFEASIBLE'], n)} / {pct(r['test_COLLISION'] + r['test_INFEASIBLE'], n)} | "
                 f"{100 * r['b_loss']:.1f} [{100 * r['b_ci95_lo']:.1f}, {100 * r['b_ci95_hi']:.1f}] | "
                 f"{f(r['c_estimate'], 4)} [{f(r['c_ci95_lo'], 4)}, {f(r['c_ci95_hi'], 4)}] |")
    L += ["", "## Analytic prediction vs observation (filter on, K = 1, 2, 3, nine actors pooled)", "",
          f"Recorded prediction: zero command admissible for a robot pair only at barrier-point separation >= "
          f"{pred['gains']['0.5']['robot_robot_zero_command_separation']:.3f} m (alpha 0.5) / "
          f"{pred['gains']['1.0']['robot_robot_zero_command_separation']:.3f} m (alpha 1); nominal spacing "
          f"{pred['nominal_centre_spacing']:.3f} m.", "",
          "| K | alpha | threshold [m] | initial min pair sep. median [m] | trials starting below threshold | nominal cmd violates a robot row (share of robot-periods) | INFEASIBLE (at step 0) | GOAL | TIMEOUT | min pair sep. median on / off [m] | interventions (share) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in an:
        L.append(f"| {r['density']} | {r['alpha']:g} | {r['zero_cmd_sep_threshold']:.3f} | {r['initial_min_bp_sep_median']:.3f} | "
                 f"{100 * r['share_initial_below_threshold']:.1f}% | {100 * r['rr_nominal_violation_frac_mean']:.1f}% | "
                 f"{r['infeasible']} ({r['infeasible_at_step0']}) | {r['goal']} | {r['timeout']} | {r['bp_sep_min_median_on']:.3f} / "
                 f"{r['bp_sep_min_median_off']:.3f} | {100 * r['interventions_frac']:.1f}% |")
    L += ["", "Barrier minima: `h_*_min_prestep` are protocol minima over pre-step states (terminal state omitted); "
          "`h_*_min_sampled` include the terminal state. Both are control-instant samples, not continuous-path bounds "
          "(columns in `endpoints.csv`)."]
    (OUT / "VALIDATION_SWEEP.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
