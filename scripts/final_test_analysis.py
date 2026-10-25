#!/usr/bin/env python3
"""Locked analysis of the FINAL TEST (docs/PREFILTER_VALIDATION_PROTOCOL_LOCKED_2026-10-25.md, D2 and
the pre-declared secondaries) at alpha* = 10. Confirmatory: per primary density, criteria (a), (b),
(c) for filter on vs filter off, pooled over the nine actors, and the usefulness verdict. Everything
else is descriptive. Reuses the comparison code of the development analysis unchanged
(scripts/validation_sweep_analysis.compare, which calls cbf_study.locked_stats).

Writes results/final_test/{endpoints.csv, verdict.json, paired_timeout_stalled.csv, FINAL_TEST.md}."""
import csv, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import validation_sweep_analysis as VA                  # noqa: E402
from cbf_study import locked_stats as LS               # noqa: E402
from cbf_study.evaluate import write_rows               # noqa: E402
from cbf_study.stats import mcnemar_exact_p, newcombe_paired_ci, paired_table   # noqa: E402

OUT = ROOT / "results" / "final_test"
A = 10.0
KS = (1, 2, 3)


def main():
    VA.OUT = OUT
    arr = VA.load()
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())
    pols = [p["policy_id"] for p in sel["policies"]]
    seeds = {s: [p["policy_id"] for p in sel["policies"] if p["seed"] == s] for s in (0, 1, 2)}
    assert all((p, K, c, a) in arr for p in pols for K in KS for c, a in (("off", None), ("on", A), ("bmatch", A))), "incomplete"
    assert all(len(arr[(p, K, "off", None)]["trial"]) == 1000 for p in pols for K in KS)
    comps = ((("off", None), ("on", A)), (("off", None), ("bmatch", A)), (("bmatch", A), ("on", A)))
    ends, verdict, tos = [], {}, []
    for K in KS:
        for ref, test in comps:
            r = VA.compare(arr, pols, K, ref, test, "pooled", "all9", A)
            ends.append(r)
            if ref[0] == "off" and test[0] == "on":
                verdict[K] = dict(
                    verdict=r["usefulness_dev"], n_pairs=r["n_pairs"],
                    a=dict(diff_on_minus_off=r["a_diff"], p_one_sided=r["a_p_one_sided"], upper_98_33=r["a_upper_9833"],
                           ci95=[r["a_ci95_lo"], r["a_ci95_hi"]], passed=bool(r["a_meets"])),
                    b=dict(goal_loss=r["b_loss"], upper_98_33=r["b_upper_9833"], ci95=[r["b_ci95_lo"], r["b_ci95_hi"]],
                           margin=LS.GOAL_MARGIN, passed=bool(r["b_meets"])),
                    c=dict(status=r["c_status"], increase_m=r["c_estimate"], upper_98_33_bootstrap=r["c_upper_9833"],
                           normal_crosscheck=r["c_normal_upper"], ci95=[r["c_ci95_lo"], r["c_ci95_hi"]], margin=LS.FORM_MARGIN,
                           eligible=r["c_eligible"], excluded=r["c_excluded"], exclusion_reasons=r["c_excl_reasons"],
                           passed=bool(r["c_meets"])))
            for s, ps in seeds.items():
                ends.append(VA.compare(arr, ps, K, ref, test, "seed", f"seed{s}", A))
            for p in pols:
                ends.append(VA.compare(arr, [p], K, ref, test, "policy", p, A))
            # TIMEOUT and stalled: paired, two-sided exact McNemar + Newcombe 95% (descriptive)
            for ep, f in (("TIMEOUT", lambda arm: VA.cat(arr, pols, K, *arm, "outcome") == "TIMEOUT"),
                          ("stalled", lambda arm: VA.cat(arr, pols, K, *arm, "stalled") == 1)):
                n11, n10, n01, n00 = paired_table(f(ref), f(test))
                e, lo, hi = newcombe_paired_ci(n11, n10, n01, n00)
                tos.append(dict(density=K, reference=ref[0], test=test[0], endpoint=ep, ref_rate=(n11 + n10) / 9000,
                                test_rate=(n11 + n01) / 9000, diff=e, ci95_lo=lo, ci95_hi=hi, mcnemar_two_sided_p=mcnemar_exact_p(n10, n01)))
        print(f"K{K}: {verdict[K]['verdict']}", flush=True)
    write_rows(OUT / "endpoints.csv", ends)
    write_rows(OUT / "paired_timeout_stalled.csv", tos)
    (OUT / "verdict.json").write_text(json.dumps(dict(
        alpha_star=A, protocol="docs/PREFILTER_VALIDATION_PROTOCOL_LOCKED_2026-10-25.md", protocol_commit="9baf6d7",
        alpha_star_commit="4fffe62", level_one_sided=LS.ALPHA_DENSITY, verdict_per_density={str(k): v for k, v in verdict.items()},
        scope="average effect over the nine fixed actors (equal weights); not arbitrary MAPPO policies, not every actor"),
        indent=1, default=float))
    write_md(ends, verdict, tos, pols, seeds)


def write_md(ends, verdict, tos, pols, seeds):
    pct = lambda k, n: f"{100 * k / n:.1f}"
    f = lambda x, p=4: "-" if x != x else f"{x:.{p}f}"
    lay = list(csv.DictReader(open(OUT / "layouts.csv")))
    meta = json.loads((OUT / "final_meta.json").read_text())
    L = ["# FINAL TEST: CBF-QP filter at alpha* = 10 vs filter off (and B-match), reserved layouts", "",
         "Protocol `docs/PREFILTER_VALIDATION_PROTOCOL_LOCKED_2026-10-25.md` (commit 9baf6d7, author-locked, **not "
         "advisor-approved**); alpha* = 10 s^-1 selected on development data by the locked rule (commit 4fffe62). Nothing "
         "in the protocol, margins, filter, scoring, tolerances, actors or config was changed. Generated by "
         "`scripts/final_test_analysis.py` from `scripts/final_test.py` runs.", "",
         f"* Data: reserved block, seed = 4,000,000 + 1,000 p + K, 1,000 layouts per actor and K = 1, 2, 3 (27 seeds, "
         f"{sum(int(r['n']) for r in lay):,} layouts), generated once; SHA-256 per array in `layouts.csv`. Block "
         "5,000,000-5,999,999 untouched.",
         "* Conditions on identical layouts: filter off, filter on (alpha* = 10), B-match (rows of alpha* = 10). "
         f"Wall time {meta['wall_s_this_invocation'] / 60:.0f} min on {meta['workers']} workers.",
         "* Scope of the inference: the average effect over these nine fixed actors with equal weights. Not arbitrary "
         "MAPPO policies, not independent training runs in general, not every individual actor.",
         "* Safety statements stay within `docs/CBF_DESIGN.md` Lemma 1 under A1-A9; these are numerical results, not a "
         "continuous-time safety claim.", "",
         "## Confirmatory verdict (per density; each component one-sided at 0.05/3; useful only if (a), (b), (c) all pass)", "",
         "| K | (a) COLLISION-or-INFEASIBLE: off -> on, diff [98.33% upper; 95% CI], exact one-sided p | (b) GOAL loss [98.33% upper; 95% CI], margin 0.05 | (c) common-prefix formation increase m [98.33% bootstrap upper; normal cross-check; 95% CI], margin 0.01 | verdict |",
         "|---|---|---|---|---|"]
    for K in KS:
        v = verdict[K]
        r = next(e for e in ends if e["scope"] == "pooled" and e["density"] == K and e["test"] == "on" and e["reference"] == "off")
        n = r["n_pairs"]
        a, b, c = v["a"], v["b"], v["c"]
        L.append(f"| {K} | {pct(r['ref_COLLISION'] + r['ref_INFEASIBLE'], n)}% -> {pct(r['test_COLLISION'] + r['test_INFEASIBLE'], n)}%, "
                 f"{100 * a['diff_on_minus_off']:.2f} pts [{100 * a['upper_98_33']:.2f}; {100 * a['ci95'][0]:.2f}, {100 * a['ci95'][1]:.2f}], "
                 f"p = {a['p_one_sided']:.2e} **{'PASS' if a['passed'] else 'FAIL'}** | "
                 f"{100 * b['goal_loss']:.2f} pts [{100 * b['upper_98_33']:.2f}; {100 * b['ci95'][0]:.2f}, {100 * b['ci95'][1]:.2f}] "
                 f"**{'PASS' if b['passed'] else 'FAIL'}** | {f(c['increase_m'])} [{f(c['upper_98_33_bootstrap'])}; {f(c['normal_crosscheck'])}; "
                 f"{f(c['ci95'][0])}, {f(c['ci95'][1])}] ({c['eligible']} eligible, {c['excluded']} excluded) "
                 f"**{c['status'].upper()}** | **{v['verdict'].upper()}** |")
    L += ["", "## Pooled endpoints (nine actors, 9,000 pairs per density)", "",
          "| K | comparison | GOAL ref / test % | COLLISION ref / test (obst/robot) | INFEASIBLE test | TIMEOUT ref / test % | stalled ref / test % | reached-goal-region ref / test % [diff 95% CI] | episode length ref / test | interventions or brake activations per trial (share of robot-periods) |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in [e for e in ends if e["scope"] == "pooled"]:
        n = r["n_pairs"]
        L.append(f"| {r['density']} | {r['test']} vs {r['reference']} | {pct(r['ref_GOAL'], n)} / {pct(r['test_GOAL'], n)} | "
                 f"{r['ref_COLLISION']} ({r['ref_coll_obstacle']}/{r['ref_coll_robot']}) / {r['test_COLLISION']} ({r['test_coll_obstacle']}/{r['test_coll_robot']}) | "
                 f"{r['test_INFEASIBLE']} | {pct(r['ref_TIMEOUT'], n)} / {pct(r['test_TIMEOUT'], n)} | {pct(r['ref_stalled'], n)} / {pct(r['test_stalled'], n)} | "
                 f"{pct(r['ref_reached_goal_region'], n)} / {pct(r['test_reached_goal_region'], n)} [{100 * r['rgr_diff']:.1f}: {100 * r['rgr_ci95_lo']:.1f}, {100 * r['rgr_ci95_hi']:.1f}] | "
                 f"{r['ref_episode_len_mean']:.0f} / {r['test_episode_len_mean']:.0f} | {r['test_interv_or_brake_per_trial']:.1f} ({100 * r['test_interv_or_brake_frac']:.1f}%) |")
    L += ["", "## Pooled secondaries", "",
          "| K | comparison | (a) diff [95%] | (b) loss [95%] | (c) increase m [95%] | final centroid distance shift: HL m [95%], Wilcoxon p | common-prefix shift: HL m [95%], Wilcoxon p | common-prefix L median [q25, q75] | both-GOAL full-trajectory formation ref / test m (pairs) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in [e for e in ends if e["scope"] == "pooled"]:
        L.append(f"| {r['density']} | {r['test']} vs {r['reference']} | {100 * r['a_diff']:.2f} [{100 * r['a_ci95_lo']:.2f}, {100 * r['a_ci95_hi']:.2f}] | "
                 f"{100 * r['b_loss']:.2f} [{100 * r['b_ci95_lo']:.2f}, {100 * r['b_ci95_hi']:.2f}] | {f(r['c_estimate'])} [{f(r['c_ci95_lo'])}, {f(r['c_ci95_hi'])}] | "
                 f"{f(r['final_dist_hl'])} [{f(r['final_dist_hl_lo'])}, {f(r['final_dist_hl_hi'])}], {r['final_dist_wilcoxon_p']:.1e} | "
                 f"{f(r.get('prefix_hl', np.nan))} [{f(r.get('prefix_hl_lo', np.nan))}, {f(r.get('prefix_hl_hi', np.nan))}], {r.get('prefix_wilcoxon_p', np.nan):.1e} | "
                 f"{r['L_median']:.0f} [{r['L_q25']:.0f}, {r['L_q75']:.0f}] | {f(r['both_goal_ref_form'])} / {f(r['both_goal_test_form'])} ({r['both_goal_pairs']}) |")
    L += ["", "Barrier minima over all 9,000 trials per arm (control-instant samples, NOT continuous-path bounds; 'pre-step' = "
          "protocol logger, terminal state omitted; 'sampled' includes the terminal state):", "",
          "| K | comparison | h_obst min pre-step / sampled, ref | test | h_rr min pre-step / sampled, ref | test |", "|---|---|---|---|---|---|"]
    for r in [e for e in ends if e["scope"] == "pooled"]:
        L.append(f"| {r['density']} | {r['test']} vs {r['reference']} | {r['ref_h_obst_min_prestep']:.4f} / {r['ref_h_obst_min_sampled']:.4f} | "
                 f"{r['test_h_obst_min_prestep']:.4f} / {r['test_h_obst_min_sampled']:.4f} | {r['ref_h_rr_min_prestep']:.4f} / {r['ref_h_rr_min_sampled']:.4f} | "
                 f"{r['test_h_rr_min_prestep']:.4f} / {r['test_h_rr_min_sampled']:.4f} |")
    L += ["", "Paired TIMEOUT and stalled (two-sided exact McNemar, Newcombe 95%; descriptive): `paired_timeout_stalled.csv`.", "",
          "## Per seed and per actor, filter on vs off (descriptive 95% intervals; no usefulness verdict per stratum)", "",
          "| K | scope | GOAL off / on % | fail off / on % | infeasible on % | (a) diff % [95%] | (b) loss % [95%] | (c) increase m [95%] | eligible / excluded | both-GOAL pairs |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in [e for e in ends if e["scope"] in ("seed", "policy") and e["test"] == "on" and e["reference"] == "off"]:
        n = r["n_pairs"]
        L.append(f"| {r['density']} | {r['scope_id']} | {pct(r['ref_GOAL'], n)} / {pct(r['test_GOAL'], n)} | "
                 f"{pct(r['ref_COLLISION'] + r['ref_INFEASIBLE'], n)} / {pct(r['test_COLLISION'] + r['test_INFEASIBLE'], n)} | {pct(r['test_INFEASIBLE'], n)} | "
                 f"{100 * r['a_diff']:.1f} [{100 * r['a_ci95_lo']:.1f}, {100 * r['a_ci95_hi']:.1f}] | {100 * r['b_loss']:.1f} [{100 * r['b_ci95_lo']:.1f}, {100 * r['b_ci95_hi']:.1f}] | "
                 f"{f(r['c_estimate'])} [{f(r['c_ci95_lo'])}, {f(r['c_ci95_hi'])}] | {r['c_eligible']} / {r['c_excluded']} | {r['both_goal_pairs']} |")
    L += ["", "## Filter on vs B-match at alpha* (same pairs; descriptive, no confirmatory claim)", "",
          "| K | scope | GOAL B-match / on % | fail B-match / on % | (a) diff on - B-match % [95%] | (b) GOAL loss of on vs B-match % [95%] | (c) increase on - B-match m [95%] |",
          "|---|---|---|---|---|---|---|"]
    for r in [e for e in ends if e["test"] == "on" and e["reference"] == "bmatch" and e["scope"] in ("pooled", "seed")]:
        n = r["n_pairs"]
        L.append(f"| {r['density']} | {r['scope_id']} | {pct(r['ref_GOAL'], n)} / {pct(r['test_GOAL'], n)} | "
                 f"{pct(r['ref_COLLISION'] + r['ref_INFEASIBLE'], n)} / {pct(r['test_COLLISION'] + r['test_INFEASIBLE'], n)} | "
                 f"{100 * r['a_diff']:.2f} [{100 * r['a_ci95_lo']:.2f}, {100 * r['a_ci95_hi']:.2f}] | {100 * r['b_loss']:.2f} [{100 * r['b_ci95_lo']:.2f}, {100 * r['b_ci95_hi']:.2f}] | "
                 f"{f(r['c_estimate'])} [{f(r['c_ci95_lo'])}, {f(r['c_ci95_hi'])}] |")
    L += ["", "## Files", "",
          "* `layouts/` (arrays), `layouts.csv` (seeds, SHA-256); `cells/*.csv.gz` (one row per trial and condition, all scalar "
          "fields); `endpoints.csv` (every comparison, pooled / per seed / per actor); `verdict.json`; `paired_timeout_stalled.csv`.",
          "* `steps/<actor>_K<K>_<condition>.npz`: per-step record of EVERY trial (ragged; `pose_offsets`, `cmd_offsets`): "
          "realised poses t = 0..T (float32, the VMAS state precision), actor raw and clipped commands (float32), applied "
          "command (float64, NaN where infeasibility prevented application), filter intervention, QP status, infeasible flag, "
          "B-match brake activation, pre-step barrier minima per robot. B-match files: `raw`/`clipped` are the actor's own "
          "commands, `applied` the B-match command."]
    (OUT / "FINAL_TEST.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
