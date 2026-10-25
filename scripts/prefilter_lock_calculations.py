#!/usr/bin/env python3
"""Calculations recorded with the locked pre-filter protocol. Arithmetic and synthetic
simulation only: no layout is sampled, no policy is loaded, no environment is stepped.

1. Analytical prediction (D4): barrier-point separation at which the zero command satisfies a
   robot-robot row,  alpha * (d^2 - D_rr^2) >= delta_ij(d) / 1  (equal split: both sides halved),
   and the analogous obstacle row, from the unchanged cbf_qp margin functions, per gain.
2. Precision scenarios (D6) for 1,000 layouts per policy and primary density (9,000 pairs):
   * paired GOAL-rate loss (one-sided 98.33% upper bound vs the 0.05 margin), normal approximation
     to the Newcombe bound with assumed discordance;
   * equal-policy common-prefix formation increase (one-sided 98.33% bound vs 0.01 m), normal
     approximation with assumed per-pair SD and exclusion share;
   * composite failure (COLLISION or INFEASIBLE), one-sided exact McNemar at 0.05/3, power by
     multinomial simulation with explicit (p10, p01).
   These are precision scenarios under stated assumptions, not guarantees of sufficiency.

Writes results/prefilter_lock/{analytic_predictions.json, precision_*.csv, PRECISION.md}."""
import json, sys
from pathlib import Path

import numpy as np
from scipy import stats as st
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "prefilter_lock"

ALPHA_D = 0.05 / 3
Z = st.norm.ppf(1 - ALPHA_D)          # one-sided 98.333% bound
N_POL, N_PER = 9, 1000
N = N_POL * N_PER


def analytic():
    from cbf_study.cbf_qp import margin_obstacle, margin_robot
    from cbf_study.config import StudyConfig
    cfg = StudyConfig.load()
    out = dict(env_fingerprint=cfg.env_fingerprint(), nominal_centre_spacing=float(cfg["scenario.slot_radius"]) * np.sqrt(3),
               note="separations of the BARRIER POINTS (offset ell ahead of the axle) at which u = 0 satisfies the row; "
                    "below them the row demands motion. Not a guaranteed minimum physical separation. With equal headings "
                    "barrier-point separation equals centre separation; with unequal headings it can differ by up to 2*ell.",
               gains={})
    for a in cfg.alphas:
        fc = cfg.filter_config(a)
        d_rr = brentq(lambda d: a * (d * d - fc.d_rr ** 2) - margin_robot(fc, d), fc.d_rr, 10.0)
        d_o = brentq(lambda d: a * (d * d - fc.d_obst ** 2) - margin_obstacle(fc, d), fc.d_obst, 10.0)
        out["gains"][str(a)] = dict(robot_robot_zero_command_separation=d_rr, obstacle_zero_command_distance=d_o,
                                    D_rr=fc.d_rr, D_o=fc.d_obst, ell=fc.ell)
    return out


def goal_scenarios():
    rows = []
    for psi in (0.05, 0.10, 0.20, 0.30):                     # P(discordant GOAL outcome)
        for loss in (0.0, 0.01, 0.02, 0.03, 0.04):           # true p_off(GOAL) - p_on(GOAL)
            if loss > psi:
                continue
            sd = np.sqrt((psi - loss ** 2) / N)
            rows.append(dict(endpoint="GOAL loss", discordance=psi, true_loss=loss, n_pairs=N,
                             expected_upper_bound=loss + Z * sd,
                             prob_bound_below_margin=float(st.norm.cdf((0.05 - loss) / sd - Z))))
    return rows


def formation_scenarios():
    rows = []
    for sd_pair in (0.005, 0.01, 0.02, 0.04):               # SD of d_i within a policy [m]
        for excl in (0.0, 0.2, 0.5, 0.8):                    # share of pairs with no common prefix
            for inc in (0.0, 0.005, 0.008, 0.009):           # true equal-policy mean increase [m]
                n_eff = N_PER * (1 - excl)
                se = np.sqrt(N_POL * sd_pair ** 2 / n_eff) / N_POL
                rows.append(dict(endpoint="common-prefix formation increase", sd_per_pair=sd_pair, exclusion_share=excl,
                                 eligible_per_policy=n_eff, true_increase=inc, se=se, expected_upper_bound=inc + Z * se,
                                 prob_bound_below_margin=float(st.norm.cdf((0.01 - inc) / se - Z))))
    return rows


def composite_scenarios(reps=4000, seed=7):
    rng = np.random.default_rng(seed)
    rows = []
    for p10 in (0.005, 0.01, 0.02, 0.05, 0.10):              # off fails, on does not
        for p01 in (0.0, 0.005, 0.01, 0.02, 0.05):           # on fails, off does not
            cnt = rng.multinomial(N, [p10, p01, 1 - p10 - p01], size=reps)
            m = cnt[:, 0] + cnt[:, 1]
            p = np.where(m > 0, st.binom.sf(cnt[:, 0] - 1, m, 0.5), 1.0)      # P(X >= n10)
            rows.append(dict(endpoint="COLLISION or INFEASIBLE", p10=p10, p01=p01, n_pairs=N,
                             net_reduction=p10 - p01, power_one_sided_exact=float((p <= ALPHA_D).mean()), reps=reps))
    return rows


def main():
    from cbf_study.evaluate import write_rows
    OUT.mkdir(parents=True, exist_ok=True)
    an = analytic()
    (OUT / "analytic_predictions.json").write_text(json.dumps(an, indent=1))
    g, f, c = goal_scenarios(), formation_scenarios(), composite_scenarios()
    write_rows(OUT / "precision_goal.csv", g)
    write_rows(OUT / "precision_formation.csv", f)
    write_rows(OUT / "precision_composite.csv", c)
    L = ["# Locked-protocol calculations (no trials)", "",
         "Generated by `scripts/prefilter_lock_calculations.py`. Arithmetic and synthetic multinomial simulation only.", "",
         "## Zero-command separations (D4 analytical prediction)", "",
         f"Nominal centre-to-centre triangle spacing {an['nominal_centre_spacing']:.3f} m. {an['note']}", "",
         "| alpha [1/s] | robot-robot barrier-point separation [m] | obstacle barrier-point distance [m] |", "|---|---|---|"]
    for a, v in an["gains"].items():
        L.append(f"| {a} | {v['robot_robot_zero_command_separation']:.4f} (D_rr = {v['D_rr']:.2f}) | "
                 f"{v['obstacle_zero_command_distance']:.4f} (D_o = {v['D_o']:.2f}) |")
    L += ["", f"## Precision plan: {N_PER} layouts per policy and primary density, {N} pairs; one-sided level 0.05/3 (z = {Z:.4f})", "",
          "Scenarios under explicit assumptions; not a guarantee of sufficiency. 'P(pass)' = probability that the bound "
          "falls inside the margin if the assumption is true (normal approximation).", "",
          "### Paired GOAL-rate loss, margin 0.05", "", "| discordance | true loss | expected 98.33% upper bound | P(pass) |", "|---|---|---|---|"]
    L += [f"| {r['discordance']} | {r['true_loss']} | {r['expected_upper_bound']:.4f} | {r['prob_bound_below_margin']:.3f} |" for r in g]
    L += ["", "### Equal-policy common-prefix formation increase, margin 0.01 m", "",
          "| SD per pair [m] | excluded share | true increase [m] | SE [m] | expected bound [m] | P(pass) |", "|---|---|---|---|---|---|"]
    L += [f"| {r['sd_per_pair']} | {r['exclusion_share']} | {r['true_increase']} | {r['se']:.5f} | {r['expected_upper_bound']:.5f} | "
          f"{r['prob_bound_below_margin']:.3f} |" for r in f]
    L += ["", "### Composite failure (COLLISION or INFEASIBLE), one-sided exact McNemar at 0.05/3", "",
          "p10 = off fails and on does not; p01 = on fails and off does not. Assumptions, not estimates; preventing a collision "
          "is not assumed to prevent infeasibility.", "", "| p10 | p01 | power |", "|---|---|---|"]
    L += [f"| {r['p10']} | {r['p01']} | {r['power_one_sided_exact']:.3f} |" for r in c]
    (OUT / "PRECISION.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
