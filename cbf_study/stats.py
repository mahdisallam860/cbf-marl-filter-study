"""
Paired statistics for filter-off vs filter-on trials on identical layouts.

Binary endpoints (COLLISION, GOAL, stalled) -> McNemar on the discordant pairs.
Continuous endpoints (formation error)     -> Wilcoxon signed-rank on the paired differences.

Conventions: for a binary event, the 2x2 table counts
    n11 both off and on, n10 off only, n01 on only, n00 neither,
and the effect is the paired risk difference  p_on - p_off = (n01 - n10) / n  (negative = the filter
lowers the event rate). For a continuous endpoint, d = x_on - x_off.

Confidence intervals:
  * single proportion: Wilson score interval;
  * paired risk difference: Newcombe (1998) method 10 (Wilson-score based, no continuity correction);
  * paired shift: Hodges-Lehmann estimator (median of Walsh averages) with the signed-rank
    (normal-approximation) interval; bootstrap percentile interval for the mean difference.

Sample size (two-sided alpha, target power):
  * McNemar: Connor (1987) closed form from the discordant probabilities, checked by simulating
    the exact McNemar test on multinomial pairs;
  * Wilcoxon: Noether (1987) closed form n = (z_{a/2} + z_b)^2 / (3 (p' - 1/2)^2),
    p' = P(D + D' > 0) estimated from the pilot, checked by resampling pilot differences and
    running the signed-rank test (simulation is the primary estimate, it handles zeros and skew).
"""
from __future__ import annotations

import numpy as np
from scipy import stats as st


# ------------------------------------------------------------------------------------------------
# Single proportions
# ------------------------------------------------------------------------------------------------
def wilson_ci(k, n, conf=0.95):
    if n == 0:
        return float("nan"), float("nan")
    z = st.norm.ppf(0.5 + conf / 2)
    p = k / n
    den = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return float(max(0.0, mid - half)), float(min(1.0, mid + half))


# ------------------------------------------------------------------------------------------------
# Paired binary endpoints
# ------------------------------------------------------------------------------------------------
def paired_table(event_off, event_on):
    a = np.asarray(event_off, dtype=bool)
    b = np.asarray(event_on, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    return int((a & b).sum()), int((a & ~b).sum()), int((~a & b).sum()), int((~a & ~b).sum())


def mcnemar_exact_p(n10, n01):
    """Two-sided exact McNemar p-value (binomial test of n10 vs n01 at 1/2). Vectorised."""
    n10 = np.asarray(n10)
    n01 = np.asarray(n01)
    m = n10 + n01
    p = np.minimum(1.0, 2.0 * st.binom.cdf(np.minimum(n10, n01), m, 0.5))
    p = np.where(m == 0, 1.0, p)
    return float(p) if p.ndim == 0 else p


def newcombe_paired_ci(n11, n10, n01, n00, conf=0.95):
    """CI for p_on - p_off (Newcombe 1998, method 10). Returns (estimate, lo, hi)."""
    n = n11 + n10 + n01 + n00
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    # proportion 1 = on, proportion 2 = off
    a, b, c, d = n11, n01, n10, n00          # a both, b on-only, c off-only, d neither
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson_ci(a + b, n, conf)
    l2, u2 = wilson_ci(a + c, n, conf)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / np.sqrt(den) if den > 0 else 0.0
    theta = p1 - p2
    dl = np.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = np.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return float(theta), float(max(-1.0, theta - dl)), float(min(1.0, theta + du))


def mcnemar_summary(event_off, event_on, conf=0.95):
    n11, n10, n01, n00 = paired_table(event_off, event_on)
    n = n11 + n10 + n01 + n00
    est, lo, hi = newcombe_paired_ci(n11, n10, n01, n00, conf)
    p10, p01 = n10 / n, n01 / n
    return dict(n=n, n11=n11, n10=n10, n01=n01, n00=n00, p_off=(n11 + n10) / n, p_on=(n11 + n01) / n,
                diff=est, diff_lo=lo, diff_hi=hi, p_value=mcnemar_exact_p(n10, n01),
                var_diff=(p10 + p01 - (p10 - p01) ** 2) / n)


def mcnemar_n_connor(p10, p01, alpha=0.05, power=0.8):
    """Pairs needed for a two-sided McNemar test (Connor 1987). p10, p01: discordant probabilities."""
    psi, delta = p10 + p01, abs(p10 - p01)
    if delta == 0:
        return float("inf")
    za, zb = st.norm.ppf(1 - alpha / 2), st.norm.ppf(power)
    n = (za * np.sqrt(psi) + zb * np.sqrt(max(psi - delta ** 2, 0.0))) ** 2 / delta ** 2
    return int(np.ceil(n))


def mcnemar_power_sim(p10, p01, n, alpha=0.05, reps=4000, seed=0):
    """Power of the exact two-sided McNemar test with n pairs (multinomial simulation)."""
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(int(n), [p10, p01, max(0.0, 1 - p10 - p01)], size=reps)
    return float((mcnemar_exact_p(cnt[:, 0], cnt[:, 1]) < alpha).mean())


# ------------------------------------------------------------------------------------------------
# Paired continuous endpoints
# ------------------------------------------------------------------------------------------------
def wilcoxon_test(d):
    """Two-sided Wilcoxon signed-rank test on paired differences d (zeros dropped, 'wilcox')."""
    d = np.asarray(d, dtype=np.float64)
    nz = int((d != 0).sum())
    if nz == 0:
        return dict(n=len(d), n_nonzero=0, statistic=float("nan"), p_value=1.0)
    r = st.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    return dict(n=len(d), n_nonzero=nz, statistic=float(r.statistic), p_value=float(r.pvalue))


def signed_rank_p_approx(D):
    """Vectorised two-sided signed-rank p-values, normal approximation, zeros dropped, no tie or
    continuity correction. D [R, n]. Used inside power simulations (cross-checked with scipy)."""
    D = np.atleast_2d(np.asarray(D, dtype=np.float64))
    nz = D != 0
    n_eff = nz.sum(axis=1)
    a = np.where(nz, np.abs(D), -1.0)                       # zeros rank lowest ...
    ranks = st.rankdata(a, axis=1) - (~nz).sum(axis=1, keepdims=True)   # ... and are removed
    tplus = np.where(nz & (D > 0), ranks, 0.0).sum(axis=1)
    mean = n_eff * (n_eff + 1) / 4.0
    var = n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24.0
    z = np.where(var > 0, (tplus - mean) / np.sqrt(np.where(var > 0, var, 1.0)), 0.0)
    return np.where(n_eff > 0, 2 * st.norm.sf(np.abs(z)), 1.0)


def hodges_lehmann(d, conf=0.95):
    """Hodges-Lehmann shift estimate (median of Walsh averages) with the signed-rank CI."""
    d = np.asarray(d, dtype=np.float64)
    n = len(d)
    i, j = np.triu_indices(n)
    w = np.sort((d[i] + d[j]) / 2.0)
    M = len(w)
    z = st.norm.ppf(0.5 + conf / 2)
    k = int(np.floor(n * (n + 1) / 4.0 - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)))
    k = min(max(k, 0), (M - 1) // 2)
    return float(np.median(w)), float(w[k]), float(w[M - 1 - k])


def bootstrap_mean_ci(d, conf=0.95, reps=4000, seed=0):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (reps, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.quantile(m, 0.5 - conf / 2)), float(np.quantile(m, 0.5 + conf / 2))


def p_prime(d):
    """P(D + D' > 0) + P(D + D' = 0)/2 over distinct pairs (Noether's p')."""
    d = np.asarray(d, dtype=np.float64)
    i, j = np.triu_indices(len(d), 1)
    s = d[i] + d[j]
    return float((s > 0).mean() + 0.5 * (s == 0).mean())


def wilcoxon_n_noether(d, alpha=0.05, power=0.8):
    pp = p_prime(d)
    if pp == 0.5:
        return float("inf")
    za, zb = st.norm.ppf(1 - alpha / 2), st.norm.ppf(power)
    return int(np.ceil((za + zb) ** 2 / (3.0 * (pp - 0.5) ** 2)))


def wilcoxon_power_sim(d_pilot, n, alpha=0.05, reps=2000, seed=0):
    """Power of the signed-rank test with n pairs, resampling the pilot differences."""
    d = np.asarray(d_pilot, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = int(n)
    chunk = max(1, min(reps, 2_000_000 // max(n, 1)))      # bound memory for large n
    hits, done = 0, 0
    while done < reps:
        m = min(chunk, reps - done)
        hits += int((signed_rank_p_approx(d[rng.integers(0, len(d), (m, n))]) < alpha).sum())
        done += m
    return hits / reps


def wilcoxon_summary(x_off, x_on, conf=0.95):
    x_off = np.asarray(x_off, dtype=np.float64)
    x_on = np.asarray(x_on, dtype=np.float64)
    d = x_on - x_off
    t = wilcoxon_test(d)
    hl, lo, hi = hodges_lehmann(d, conf)
    m, mlo, mhi = bootstrap_mean_ci(d, conf)
    return dict(n=t["n"], n_nonzero=t["n_nonzero"], mean_off=float(x_off.mean()), mean_on=float(x_on.mean()),
                mean_diff=m, mean_diff_lo=mlo, mean_diff_hi=mhi, sd_diff=float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                hl=hl, hl_lo=lo, hl_hi=hi, p_value=t["p_value"], p_prime=p_prime(d))


# ------------------------------------------------------------------------------------------------
# Smallest n reaching the target power
# ------------------------------------------------------------------------------------------------
def min_n_for_power(power_fn, target=0.8, n_start=10, n_max=200000):
    """Smallest n with power_fn(n) >= target, by doubling then bisection. power_fn should use
    common random numbers (a fixed seed) so it is close to monotone. Returns inf beyond n_max."""
    lo, hi = 1, max(2, int(n_start))
    while power_fn(hi) < target:
        lo, hi = hi, hi * 2
        if hi > n_max:
            return float("inf")
    while hi - lo > max(1, lo // 50):
        mid = (lo + hi) // 2
        if power_fn(mid) >= target:
            hi = mid
        else:
            lo = mid
    return int(hi)
