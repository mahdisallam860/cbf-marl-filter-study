"""Locked confirmatory statistics: sign conventions, one-sided levels, equal policy weights,
inconclusive handling. Synthetic data only."""
import tomllib
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as st

from cbf_study import locked_stats as L
from cbf_study.stats import newcombe_paired_ci

ROOT = Path(__file__).resolve().parents[1]
LOCK = tomllib.loads((ROOT / "config" / "evaluation_protocol_locked.toml").read_text())["confirmatory"]
POL = [f"p{i}" for i in range(9)]


def test_constants_match_locked_config():
    assert L.ALPHA_DENSITY == LOCK["alpha_density"] and L.GOAL_MARGIN == LOCK["goal_loss_margin"]
    assert L.FORM_MARGIN == LOCK["formation_margin_m"] and L.BOOT_REPS == LOCK["bootstrap_replicates"]
    assert L.BOOT_SEED == LOCK["bootstrap_seed"] and L.MIN_ELIGIBLE == LOCK["min_eligible_per_policy"]
    assert L.DESCRIPTIVE_CONF == LOCK["descriptive_conf"]


def pairs(n10, n01, n11, n00, per_policy=True):
    off = np.r_[np.ones(n10), np.zeros(n01), np.ones(n11), np.zeros(n00)].astype(bool)
    on = np.r_[np.zeros(n10), np.ones(n01), np.ones(n11), np.zeros(n00)].astype(bool)
    n = len(off)
    pol = np.repeat(POL, n // 9) if per_policy else np.array(["p0"] * n)
    return off, on, pol


def test_failure_reduction_sign_one_sided_exact_p_and_level():
    off, on, pol = pairs(60, 15, 30, 795)                     # 900 pairs, filter fails less
    r = L.failure_reduction(off, on, pol)
    assert r["diff_on_minus_off"] == pytest.approx((15 - 60) / 900) and r["diff_on_minus_off"] < 0
    assert r["p_one_sided"] == pytest.approx(st.binom.sf(59, 75, 0.5)) and r["passed"]
    est, lo, hi = newcombe_paired_ci(30, 60, 15, 795, conf=1 - 2 * (0.05 / 3))
    assert r["upper_one_sided"] == pytest.approx(hi)
    assert r["ci95"][1] < r["upper_one_sided"]                 # 95 % interval is narrower
    worse = L.failure_reduction(on, off, pol)                  # filter fails MORE: one-sided p near 1
    assert worse["p_one_sided"] > 0.99 and not worse["passed"]
    assert L.mcnemar_one_sided_p(0, 0) == 1.0


def test_goal_loss_sign_margin_and_infeasible_arm_counts_as_non_goal():
    # 900 pairs: 54 lose GOAL with the filter (e.g. INFEASIBLE arm -> GOAL = 0), 27 gain, others concordant
    off, on, pol = pairs(54, 27, 600, 219)
    r = L.goal_noninferiority(off, on, pol)
    assert r["loss"] == pytest.approx(27 / 900)               # positive = deterioration
    est, lo, hi = newcombe_paired_ci(600, 54, 27, 219, conf=1 - 2 * (0.05 / 3))
    assert r["upper_one_sided"] == pytest.approx(-lo)
    assert r["passed"] == (r["upper_one_sided"] <= 0.05)
    big = L.goal_noninferiority(*pairs(108, 0, 600, 192))     # 12 % loss: must fail the 0.05 margin
    assert not big["passed"] and big["loss"] == pytest.approx(0.12)


def test_unequal_allocation_is_refused():
    off, on, _ = pairs(10, 5, 10, 11)
    with pytest.raises(ValueError):
        L.failure_reduction(off, on, np.array(["a"] * 20 + ["b"] * 16))


def test_formation_equal_policy_weights_with_unequal_eligible_counts():
    d, elig, pol = [], [], []
    for i, p in enumerate(POL):                                # policy i: 2 + 50 i eligible pairs, all equal to i/1000
        m = 2 + 50 * i
        d += [i / 1000] * m + [np.nan] * 3
        elig += [True] * m + [False] * 3
        pol += [p] * (m + 3)
    r = L.formation_noninferiority(np.array(d), np.array(elig), np.array(pol), POL, reps=2000)
    assert r["estimate"] == pytest.approx(np.mean([i / 1000 for i in range(9)]))   # not the pair-weighted mean
    assert r["excluded_per_policy"]["p0"] == 3 and r["eligible_per_policy"]["p8"] == 402
    assert r["upper_one_sided"] == pytest.approx(r["estimate"]) and r["passed"]     # no within-policy spread


def test_formation_bootstrap_is_seeded_and_bound_is_one_sided_quantile():
    rng = np.random.default_rng(3)
    pol = np.repeat(POL, 100)
    d = rng.normal(0.004, 0.02, 900)
    elig = np.ones(900, bool)
    a = L.formation_noninferiority(d, elig, pol, POL, reps=4000)
    b = L.formation_noninferiority(d, elig, pol, POL, reps=4000)
    assert a["upper_one_sided"] == b["upper_one_sided"]
    assert a["ci95"][0] < a["estimate"] < a["ci95"][1] < a["upper_one_sided"]
    assert a["upper_one_sided"] == pytest.approx(a["normal_upper_crosscheck"], abs=5e-4)
    shifted = L.formation_noninferiority(d + 0.02, elig, pol, POL, reps=4000)
    assert not shifted["passed"]


def test_policy_without_eligible_pairs_makes_c_inconclusive():
    pol = np.repeat(POL, 10)
    d = np.zeros(90)
    elig = np.ones(90, bool)
    elig[pol == "p4"] = False
    r = L.formation_noninferiority(d, elig, pol, POL)
    assert r["status"] == "inconclusive" and "p4" in r["reason"] and not r["passed"]
    a = dict(passed=True)
    assert L.usefulness(a, a, r) == "inconclusive"
    assert L.usefulness(a, dict(passed=False), r) == "not shown useful"
    assert L.usefulness(a, a, dict(passed=True, status="pass")) == "useful"
