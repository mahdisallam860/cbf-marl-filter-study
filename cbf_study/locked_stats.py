"""
Confirmatory statistics of the author-locked protocol (D2), per primary density.

Sign conventions:
  (a) failure F = COLLISION or INFEASIBLE;  diff = P(F_on) - P(F_off)  (negative = fewer failures).
      One-sided exact McNemar, H1: filter on fails less often: p = P(X >= n10), X ~ Bin(n10+n01, 1/2),
      n10 = #(F_off & ~F_on). Pass: p <= ALPHA_DENSITY.
  (b) GOAL loss = P(GOAL_off) - P(GOAL_on)  (positive = deterioration). One-sided upper bound at
      1 - ALPHA_DENSITY = -lower limit of the two-sided (1 - 2 ALPHA_DENSITY) Newcombe method-10
      interval for P(GOAL_on) - P(GOAL_off). Pass: bound <= GOAL_MARGIN.
  (c) formation increase = equal-weight mean over policies of the mean eligible common-prefix
      difference d (on - off). Upper bound = quantile 1 - ALPHA_DENSITY (numpy 'linear') of a
      stratified paired percentile bootstrap (resample eligible pairs within each policy, fixed
      counts, weights 1/n_policies; B = BOOT_REPS, default_rng(BOOT_SEED)). Pass: bound <= FORM_MARGIN.
      A policy with fewer than MIN_ELIGIBLE eligible pairs makes (c) inconclusive (no renormalising).
Useful at a density only if (a), (b) and (c) all pass (intersection-union; no further split).
Every helper also returns an ordinary two-sided 95 % interval (descriptive).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st

from .stats import newcombe_paired_ci, paired_table

ALPHA_DENSITY = 0.05 / 3
GOAL_MARGIN = 0.05
FORM_MARGIN = 0.01
BOOT_REPS = 20000
BOOT_SEED = 20261026
MIN_ELIGIBLE = 2
DESCRIPTIVE_CONF = 0.95


def check_equal_allocation(policy):
    """Pooling gives equal policy weights only with equal pair counts; refuse otherwise."""
    _, counts = np.unique(np.asarray(policy), return_counts=True)
    if len(counts) == 0 or counts.min() != counts.max():
        raise ValueError(f"unequal pairs per policy {counts.tolist()}: equal allocation is required")
    return int(counts[0])


def mcnemar_one_sided_p(n10, n01):
    """P(X >= n10), X ~ Binomial(n10 + n01, 1/2); 1 when there are no discordant pairs."""
    m = n10 + n01
    return 1.0 if m == 0 else float(st.binom.sf(n10 - 1, m, 0.5))


def failure_reduction(fail_off, fail_on, policy, alpha=ALPHA_DENSITY):
    check_equal_allocation(policy)
    n11, n10, n01, n00 = paired_table(fail_off, fail_on)
    p = mcnemar_one_sided_p(n10, n01)
    est, _, hi = newcombe_paired_ci(n11, n10, n01, n00, conf=1 - 2 * alpha)
    _, lo95, hi95 = newcombe_paired_ci(n11, n10, n01, n00, conf=DESCRIPTIVE_CONF)
    return dict(n=n11 + n10 + n01 + n00, n10=n10, n01=n01, diff_on_minus_off=est, upper_one_sided=hi,
                ci95=(lo95, hi95), p_one_sided=p, passed=bool(p <= alpha))


def goal_noninferiority(goal_off, goal_on, policy, margin=GOAL_MARGIN, alpha=ALPHA_DENSITY):
    """Every pair kept; an INFEASIBLE arm simply has GOAL = 0."""
    check_equal_allocation(policy)
    n11, n10, n01, n00 = paired_table(goal_off, goal_on)       # n10: GOAL off only (a loss)
    est_on_off, lo, _ = newcombe_paired_ci(n11, n10, n01, n00, conf=1 - 2 * alpha)
    _, lo95, hi95 = newcombe_paired_ci(n11, n10, n01, n00, conf=DESCRIPTIVE_CONF)
    loss, upper = -est_on_off, -lo
    return dict(n=n11 + n10 + n01 + n00, goal_off_only=n10, goal_on_only=n01, loss=loss, upper_one_sided=upper,
                ci95_loss=(-hi95, -lo95), margin=margin, passed=bool(upper <= margin))


def _policy_groups(d, eligible, policy, policies):
    d, eligible, policy = np.asarray(d, float), np.asarray(eligible, bool), np.asarray(policy)
    return [d[(policy == p) & eligible] for p in policies]


def formation_noninferiority(d, eligible, policy, policies, margin=FORM_MARGIN, alpha=ALPHA_DENSITY,
                             reps=BOOT_REPS, seed=BOOT_SEED):
    """d [n] common-prefix differences (NaN for excluded), eligible [n], policy [n] labels,
    policies = the full fixed list (a policy with no eligible pair is NOT dropped)."""
    groups = _policy_groups(d, eligible, policy, policies)
    counts = {str(p): len(g) for p, g in zip(policies, groups)}
    excluded = {str(p): int(((np.asarray(policy) == p) & ~np.asarray(eligible, bool)).sum()) for p in policies}
    short = [str(p) for p, g in zip(policies, groups) if len(g) < MIN_ELIGIBLE]
    base = dict(eligible_per_policy=counts, excluded_per_policy=excluded, margin=margin)
    if short:
        return dict(base, status="inconclusive", reason=f"policies with fewer than {MIN_ELIGIBLE} eligible pairs: {short}",
                    estimate=float("nan"), upper_one_sided=float("nan"), passed=False)
    P = len(groups)
    means = np.array([g.mean() for g in groups])
    est = float(means.mean())
    rng = np.random.default_rng(seed)
    boot = np.zeros(reps)
    for g in groups:
        done = 0
        while done < reps:                                   # chunks bound the index array size
            m = min(reps - done, max(1, 2_000_000 // len(g)))
            boot[done:done + m] += g[rng.integers(0, len(g), (m, len(g)))].mean(axis=1) / P
            done += m
    upper = float(np.quantile(boot, 1 - alpha, method="linear"))
    lo95, hi95 = (float(x) for x in np.quantile(boot, [0.025, 0.975], method="linear"))
    z = st.norm.ppf(1 - alpha)
    se = float(np.sqrt(sum(g.var(ddof=1) / len(g) for g in groups)) / P)
    normal_upper = est + z * se
    passed = upper <= margin
    return dict(base, status="pass" if passed else "fail", estimate=est, policy_means=dict(zip(map(str, policies), means.tolist())),
                upper_one_sided=upper, ci95=(lo95, hi95), normal_upper_crosscheck=normal_upper,
                crosscheck_disagrees=bool((normal_upper <= margin) != passed), passed=bool(passed), reps=reps, seed=seed)


def usefulness(a, b, c):
    """'useful' only if (a), (b), (c) all pass; 'inconclusive' if (c) is inconclusive and (a), (b) pass."""
    if a["passed"] and b["passed"] and c["passed"]:
        return "useful"
    if a["passed"] and b["passed"] and c.get("status") == "inconclusive":
        return "inconclusive"
    return "not shown useful"
