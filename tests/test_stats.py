"""stats.py on synthetic data: exact values, agreement with scipy, CI coverage, and sample sizes
that reach the target power in simulation."""
import numpy as np
import pytest
from scipy import stats as st

from cbf_study import stats as S


def test_wilson_matches_scipy():
    for k, n in ((0, 20), (3, 20), (20, 20), (37, 200)):
        lo, hi = S.wilson_ci(k, n)
        ref = st.binomtest(k, n).proportion_ci(method="wilson")
        assert np.isclose(lo, ref.low) and np.isclose(hi, ref.high)


def test_paired_table_and_mcnemar_exact_values():
    off = np.array([1, 1, 1, 0, 0, 1, 0, 1], bool)
    on = np.array([1, 0, 0, 0, 1, 0, 0, 0], bool)
    assert S.paired_table(off, on) == (1, 4, 1, 2)
    # b=10, c=2: p = 2 * P(X <= 2 | 12, 1/2) = 2 * 79 / 4096
    assert np.isclose(S.mcnemar_exact_p(10, 2), 2 * 79 / 4096)
    assert S.mcnemar_exact_p(5, 5) == 1.0 and S.mcnemar_exact_p(0, 0) == 1.0
    assert S.mcnemar_exact_p(2, 10) == S.mcnemar_exact_p(10, 2)
    for b, c in ((7, 1), (30, 12), (0, 9)):
        assert np.isclose(S.mcnemar_exact_p(b, c), st.binomtest(b, b + c, 0.5).pvalue)


def _paired_sample(rng, n, p_off, p_on, rho):
    z = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n)
    return z[:, 0] < st.norm.ppf(p_off), z[:, 1] < st.norm.ppf(p_on)


def test_newcombe_ci_coverage_and_large_sample_agreement():
    rng = np.random.default_rng(0)
    p_off, p_on, rho, n = 0.30, 0.12, 0.4, 150
    # true paired difference by simulation of the latent model
    big_off, big_on = _paired_sample(rng, 2_000_000, p_off, p_on, rho)
    truth = big_on.mean() - big_off.mean()
    cover = 0
    reps = 1500
    for _ in range(reps):
        off, on = _paired_sample(rng, n, p_off, p_on, rho)
        est, lo, hi = S.newcombe_paired_ci(*S.paired_table(off, on))
        assert -1 <= lo <= est <= hi <= 1
        cover += lo <= truth <= hi
    assert 0.93 <= cover / reps <= 0.97
    # large n: close to the Wald interval for a paired difference
    off, on = _paired_sample(rng, 20000, p_off, p_on, rho)
    t = S.paired_table(off, on)
    est, lo, hi = S.newcombe_paired_ci(*t)
    n10, n01, nn = t[1], t[2], sum(t)
    se = np.sqrt((n10 + n01) / nn - ((n01 - n10) / nn) ** 2) / np.sqrt(nn)
    assert abs(lo - (est - 1.96 * se)) < 2e-3 and abs(hi - (est + 1.96 * se)) < 2e-3


def test_connor_sample_size_reaches_target_power():
    for p10, p01 in ((0.15, 0.03), (0.10, 0.04), (0.25, 0.10)):
        n = S.mcnemar_n_connor(p10, p01)
        pw = S.mcnemar_power_sim(p10, p01, n, reps=6000, seed=1)
        assert 0.76 <= pw <= 0.88, (p10, p01, n, pw)
        assert S.mcnemar_power_sim(p10, p01, n // 2, reps=6000, seed=1) < 0.75
    assert S.mcnemar_n_connor(0.1, 0.1) == float("inf")
    # simulation-based search agrees with the closed form
    n_sim = S.min_n_for_power(lambda n: S.mcnemar_power_sim(0.15, 0.03, n, reps=4000, seed=2))
    assert abs(n_sim - S.mcnemar_n_connor(0.15, 0.03)) <= 0.15 * n_sim


def test_vectorised_signed_rank_matches_scipy():
    rng = np.random.default_rng(3)
    D = rng.normal(0.2, 1.0, (40, 60))
    D[:, :7] = 0.0                                          # zeros are dropped, as in the 'wilcox' method
    p = S.signed_rank_p_approx(D)
    ref = np.array([st.wilcoxon(r, zero_method="wilcox", correction=False, method="approx").pvalue for r in D])
    assert np.allclose(p, ref, rtol=1e-8, atol=1e-12)
    assert S.signed_rank_p_approx(np.zeros((1, 10)))[0] == 1.0


def test_hodges_lehmann_recovers_shift_with_correct_coverage():
    rng = np.random.default_rng(4)
    shift, n, reps, cover = 0.3, 80, 800, 0
    for _ in range(reps):
        d = shift + rng.standard_t(4, n) * 0.5
        hl, lo, hi = S.hodges_lehmann(d)
        assert lo <= hl <= hi
        cover += lo <= shift <= hi
    assert 0.93 <= cover / reps <= 0.97
    d = shift + rng.normal(0, 0.5, 5000)
    assert abs(S.hodges_lehmann(d)[0] - shift) < 0.03


def test_noether_sample_size_reaches_target_power():
    rng = np.random.default_rng(5)
    for effect in (0.3, 0.5):
        pilot = rng.normal(effect, 1.0, 4000)
        n = S.wilcoxon_n_noether(pilot)
        # p' for a normal sample: Phi(sqrt(2) * mean / sd), evaluated at the sample's own moments
        assert abs(S.p_prime(pilot) - st.norm.cdf(np.sqrt(2) * pilot.mean() / pilot.std())) < 0.005
        pw = S.wilcoxon_power_sim(pilot, n, reps=3000, seed=6)
        assert 0.76 <= pw <= 0.90, (effect, n, pw)
        n_sim = S.min_n_for_power(lambda m: S.wilcoxon_power_sim(pilot, m, reps=2000, seed=7))
        assert abs(n_sim - n) <= 0.2 * n, (n_sim, n)
    assert S.wilcoxon_n_noether(np.r_[np.ones(10), -np.ones(10)]) == float("inf")


def test_summaries_on_a_synthetic_paired_study():
    rng = np.random.default_rng(8)
    n = 300
    off, on = _paired_sample(rng, n, 0.30, 0.10, 0.3)
    m = S.mcnemar_summary(off, on)
    assert m["n"] == n and m["diff"] < 0 and m["diff_hi"] < 0 and m["p_value"] < 1e-4
    x_off = rng.gamma(2.0, 0.03, n)
    x_on = x_off + np.where(rng.random(n) < 0.4, rng.gamma(2.0, 0.01, n), 0.0)   # many exact zeros
    w = S.wilcoxon_summary(x_off, x_on)
    assert w["n_nonzero"] < n and w["mean_diff"] > 0 and w["p_value"] < 1e-6
    assert w["mean_diff_lo"] <= w["mean_diff"] <= w["mean_diff_hi"]
