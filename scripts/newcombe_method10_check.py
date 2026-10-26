#!/usr/bin/env python3
"""Disclosure check, added 2026-10-26 after the final test (exploratory, post hoc).

The locked protocol and cbf_study/stats.py describe the paired interval for criteria (a) and (b) as
Newcombe (1998) method 10, but newcombe_paired_ci uses the uncorrected correlation phi, which is
Newcombe's method 8 (Wilson score limits, phi uncorrected). Method 10 replaces the numerator of phi by
max(eh - fg - n/2, 0) when eh > fg. This script recomputes the one-sided 98.33% bounds with method 10
from the per-trial final-test results so the effect on the verdict can be reported. It does not change
any confirmatory number.

Writes results/paper_revision/newcombe_method10_check.csv.
"""
import csv, gzip, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbf_study.stats import wilson_ci, newcombe_paired_ci   # noqa: E402

FT = ROOT / "results" / "final_test"
LEVEL = 1 - 2 * (0.05 / 3)          # two-sided 96.67% -> one-sided 98.33%


def newcombe_m10(a, b, c, d, conf):
    """p_on - p_off; a both, b on-only, c off-only, d neither (same convention as stats.newcombe_paired_ci)."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson_ci(a + b, n, conf)
    l2, u2 = wilson_ci(a + c, n, conf)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    num = a * d - b * c
    if num > 0:
        num = max(num - n / 2, 0.0)
    phi = num / np.sqrt(den) if den > 0 else 0.0
    th = p1 - p2
    dl = np.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = np.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return th, max(-1.0, th - dl), min(1.0, th + du)


def main():
    rows = []
    for K in (1, 2, 3):
        off, on = {}, {}
        for p in sorted((FT / "cells").glob(f"*_K{K}.csv.gz")):
            for r in csv.DictReader(gzip.open(p, "rt")):
                key = (r["policy_id"], r["trial"])
                if r["condition"] == "off":
                    off[key] = r
                elif r["condition"] == "on":
                    on[key] = r
        keys = sorted(off)
        assert set(keys) == set(on)
        for ep, field in (("a_fail", "fail"), ("b_goal", "goal")):
            x_off = np.array([int(off[k][field]) for k in keys], bool)
            x_on = np.array([int(on[k][field]) for k in keys], bool)
            a = int((x_on & x_off).sum()); b = int((x_on & ~x_off).sum()); c = int((~x_on & x_off).sum()); d = int((~x_on & ~x_off).sum())
            m8 = newcombe_paired_ci(a, c, b, d, conf=LEVEL)            # stats.py signature: (n11, n10, n01, n00)
            m10 = newcombe_m10(a, b, c, d, LEVEL)
            if ep == "a_fail":      # one-sided UPPER bound of on - off (criterion: < 0, McNemar decides)
                b8, b10 = m8[2], m10[2]
            else:                   # GOAL loss = off - on; its upper bound = -(lower bound of on - off)
                b8, b10 = -m8[1], -m10[1]
            rows.append(dict(density=K, endpoint=ep, n=len(keys), estimate_on_minus_off=m8[0],
                             bound_9833_method8_as_implemented=b8, bound_9833_method10=b10,
                             margin="" if ep == "a_fail" else 0.05,
                             meets_method8=(b8 < 0) if ep == "a_fail" else (b8 <= 0.05),
                             meets_method10=(b10 < 0) if ep == "a_fail" else (b10 <= 0.05)))
    out = ROOT / "results" / "paper_revision" / "newcombe_method10_check.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
