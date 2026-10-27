#!/usr/bin/env python3
"""Independent check of results/paper_revision/{on_vs_bmatch.csv, lost_gained.csv, transitions_4x4.csv}.

Does not import cbf_study, scripts/paper_bmatch_transitions.py or any project statistics helper. It has
its own Wilson limits (normal quantile from the standard library), Newcombe methods 10 and 8, its own
formation error with slots built from the protocol geometry (equilateral triangle, radius 0.30 m,
angles pi/2 + 2*pi*r/3) and its own implementation of the locked bootstrap draw order. Outcomes are
taken from the NPZ outcome arrays (not the cell CSVs). The reached-goal-region flags are the saved
flags from the cell CSVs: only their aggregation is verified, not the flags themselves.

Integers are compared exactly, interval and rate floats to 1e-12, formation values to 1e-8 m, the
bootstrap quantiles bit-identically or, failing that, to the Monte-Carlo tolerance 2e-4 m.
Writes results/paper_revision/bmatch_transitions_verification.md; prints PASS/FAIL and the check count.
"""
import csv, gzip, json, math, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "results" / "final_test"
RP = ROOT / "results" / "paper_revision"
CODES = {1: "GOAL", 2: "COLLISION", 3: "INFEASIBLE", 4: "TIMEOUT"}
EXPECT = ["s0_early", "s0_middle", "s0_late", "s1_early", "s1_middle", "s1_late", "s2_early", "s2_middle", "s2_late"]

checks = Counter()
fails = []


def check(ok, what):
    checks["n"] += 1
    if not ok:
        fails.append(what)


def close(x, y, tol, what):
    check(abs(float(x) - float(y)) <= tol, f"{what}: {x} vs {y}")


def wilson(k, n, z):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = (z / (1 + z * z / n)) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, centre - half), min(1.0, centre + half)


def newcombe(a, b, c, d, method):
    """Difference in proportions, arm 1 (a+b) minus arm 2 (a+c). method 10 or 8."""
    z = statistics.NormalDist().inv_cdf(0.975)
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    marg = [a + b, c + d, a + c, b + d]
    if min(marg) == 0:
        phi = 0.0
    else:
        top = a * d - b * c
        if method == 10 and top > 0:
            top = max(top - n / 2.0, 0.0)
        phi = top / math.sqrt(marg[0] * marg[1] * marg[2] * marg[3])
    diff = p1 - p2
    lo = diff - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = diff + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return diff, max(-1.0, lo), min(1.0, hi)


SLOTS = np.array([[0.30 * math.cos(math.pi / 2 + 2 * math.pi * r / 3), 0.30 * math.sin(math.pi / 2 + 2 * math.pi * r / 3)] for r in range(3)])


def ferr(poses):
    xy = np.asarray(poses, dtype=np.float64)[:, :, :2]
    return np.mean(np.linalg.norm(xy - xy.mean(axis=1)[:, None, :] - SLOTS[None], axis=2), axis=1)


def locked_bootstrap(groups, reps=20000, seed=20261026):
    gen = np.random.default_rng(seed)
    acc = np.zeros(reps)
    for g in groups:
        filled = 0
        chunk = max(1, 2_000_000 // len(g))
        while filled < reps:
            m = min(reps - filled, chunk)
            idx = gen.integers(0, len(g), (m, len(g)))
            acc[filled:filled + m] += g[idx].mean(axis=1) / len(groups)
            filled += m
    return np.quantile(acc, [0.025, 0.975], method="linear")


def main():
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())["policies"]
    actors = [p["policy_id"] for p in sel]
    check(actors == EXPECT, "actor order")
    seed = {p["policy_id"]: int(p["seed"]) for p in sel}
    check(all(seed[a] == int(a[1]) for a in actors), "seed mapping")
    check(np.allclose(SLOTS, np.load(FT / "layouts/s0_early_K1.npz")["slots"], atol=1e-12, rtol=0), "slot geometry")
    outc, form, rgr, keys = {}, {}, {}, {}
    for K in (1, 2, 3):
        for a in actors:
            seen = set()
            hashes = defaultdict(dict)
            rg = {}
            for row in csv.DictReader(gzip.open(FT / f"cells/{a}_K{K}.csv.gz", "rt")):
                k = (row["condition"], int(row["trial"]))
                check(k not in seen, f"duplicate {a} K{K} {k}")
                seen.add(k)
                check(int(row["seed"]) == seed[a], "seed in cell")
                hashes[k[1]][k[0]] = row["layout_hash"]
                if row["condition"] == "on":
                    rg[k[1]] = int(row["reached_goal_region"])
            check(len(seen) == 3000 and set(hashes) == set(range(1000)), f"trials {a} K{K}")
            check(all(len(set(h.values())) == 1 and len(h) == 3 for h in hashes.values()), f"layout hashes {a} K{K}")
            rgr[(K, a)] = np.array([rg[t] for t in range(1000)])
            e = {}
            for arm in ("off", "on", "bmatch"):
                z = dict(np.load(FT / f"steps/{a}_K{K}_{arm}.npz"))
                outc[(K, a, arm)] = np.array([CODES[int(x)] for x in z["outcome"]])
                check(len(outc[(K, a, arm)]) == 1000, "npz length")
                if arm != "off":
                    po, T = z["pose_offsets"], z["executed_periods"]
                    check(bool(np.all(po[1:] - po[:-1] == T + 1)), "pose offsets")
                    e[arm] = (ferr(z["poses"]), po, T)
            d = np.full(1000, np.nan)
            (eo, po_o, To), (eb, po_b, Tb) = e["on"], e["bmatch"]
            for t in range(1000):
                L = int(min(To[t], Tb[t]))
                if L > 0:
                    d[t] = np.mean(eo[po_o[t] + 1:po_o[t] + 1 + L] - eb[po_b[t] + 1:po_b[t] + 1 + L])
            form[(K, a)] = d

    # ---------------- on vs B-match
    csv1 = list(csv.DictReader(open(RP / "on_vs_bmatch.csv")))
    check(len(csv1) == 3 * 4 * 13, "on_vs_bmatch row count")
    boot_mode, boot_maxdiff = None, 0.0
    for r in csv1:
        K, sc, sid = int(r["density"]), r["scope"], r["scope_id"]
        acts = actors if sc == "pooled" else [a for a in actors if f"seed{seed[a]}" == sid] if sc == "seed" else [sid]
        check(len(acts) in (1, 3, 9), "scope actors")
        check(int(r["n"]) == 1000 * len(acts), "n")
        check(r["label"] == "pre-declared secondary, descriptive", "label 1")
        if r["endpoint"] == "FORMATION_M":
            gs = [form[(K, a)][np.isfinite(form[(K, a)])] for a in acts]
            check(int(r["eligible"]) == sum(len(g) for g in gs) and int(r["excluded"]) == 1000 * len(acts) - sum(len(g) for g in gs), "eligible")
            close(r["estimate"], np.mean([g.mean() for g in gs]), 1e-8, f"formation {K} {sid}")
            if sc == "pooled":
                lo, hi = locked_bootstrap(gs)
                if float(r["boot_lo"]) == lo and float(r["boot_hi"]) == hi:
                    boot_mode = boot_mode or "bit-identical"
                else:
                    boot_mode = "Monte-Carlo tolerance 2e-4 m"
                boot_maxdiff = max(boot_maxdiff, abs(float(r["boot_lo"]) - lo), abs(float(r["boot_hi"]) - hi))
                close(r["boot_lo"], lo, 2e-4, "boot lo"); close(r["boot_hi"], hi, 2e-4, "boot hi")
            else:
                check(r["boot_lo"] == r["boot_hi"] == "", "no bootstrap outside pooled")
            continue
        ev = {"GOAL": ("GOAL",), "FAIL": ("COLLISION", "INFEASIBLE"), "TIMEOUT": ("TIMEOUT",)}[r["endpoint"]]
        x = np.concatenate([np.isin(outc[(K, a, "on")], ev) for a in acts])
        y = np.concatenate([np.isin(outc[(K, a, "bmatch")], ev) for a in acts])
        cnt = [int(np.sum(x & y)), int(np.sum(x & ~y)), int(np.sum(~x & y)), int(np.sum(~x & ~y))]
        check([int(r[c]) for c in "abcd"] == cnt, f"2x2 {K} {sid} {r['endpoint']}")
        close(r["rate_on"], x.mean(), 1e-12, "rate on"); close(r["rate_bmatch"], y.mean(), 1e-12, "rate bm")
        for m, cols in ((10, ("m10_lo", "m10_hi")), (8, ("m8_lo", "m8_hi"))):
            dd, lo, hi = newcombe(*cnt, m)
            close(r["estimate"], dd, 1e-12, "estimate")
            close(r[cols[0]], lo, 1e-12, f"{K} {sid} {r['endpoint']} m{m} lo"); close(r[cols[1]], hi, 1e-12, f"m{m} hi")
            check(math.isfinite(lo) and -1 <= lo <= dd <= hi <= 1, "bounds ordered")
    # constructed fixtures for this implementation
    for case in [(0, 0, 0, 50), (50, 0, 0, 0), (20, 0, 0, 30), (10, 7, 0, 33), (0, 25, 20, 5)]:
        for m in (10, 8):
            dd, lo, hi = newcombe(*case, m)
            rd, rlo, rhi = newcombe(case[0], case[2], case[1], case[3], m)
            check(math.isfinite(lo) and math.isfinite(hi) and -1 <= lo <= dd <= hi <= 1 and hi > lo, f"fixture {case}")
            check(abs(rd + dd) < 1e-12 and abs(rlo + hi) < 1e-12 and abs(rhi + lo) < 1e-12, f"reversal {case}")

    # ---------------- lost / gained
    tc = {}
    for r in csv.DictReader(open(FT / "mechanism/timeout_classes.csv")):
        if r["condition"] == "on":
            k = (r["policy_id"], int(r["density"]), int(r["trial"]))
            check(k not in tc, "timeout class duplicate")
            tc[k] = r["category"]
    on_to = {(a, K, t) for K in (1, 2, 3) for a in actors for t in np.nonzero(outc[(K, a, "on")] == "TIMEOUT")[0].tolist()}
    check(set(tc) == on_to, "timeout class coverage")
    verdict = json.loads((FT / "verdict.json").read_text())["verdict_per_density"]

    def tally(cells):
        L, G, RG = Counter(), Counter(), Counter()
        net = 0
        for K, a in cells:
            for t, (o1, o2) in enumerate(zip(outc[(K, a, "off")], outc[(K, a, "on")])):
                net += (o1 == "GOAL") - (o2 == "GOAL")
                if o1 == "GOAL" and o2 != "GOAL":
                    L["total"] += 1
                    RG[int(rgr[(K, a)][t])] += 1
                    if o2 == "TIMEOUT":
                        cat = tc[(a, K, t)]
                        if cat.startswith("filter-induced stall: "):
                            L["filter_induced_stall"] += 1
                            L["filter_induced_stall:" + cat[len("filter-induced stall: "):]] += 1
                        else:
                            L["other_timeout"] += 1
                            L["other_timeout:" + cat.replace(" ", "_").replace("-", "_")] += 1
                    else:
                        L[o2.lower()] += 1
                elif o1 != "GOAL" and o2 == "GOAL":
                    G["total"] += 1
                    G["from_" + o1.lower()] += 1
        return L, G, RG, net

    csv2 = list(csv.DictReader(open(RP / "lost_gained.csv")))
    groups2 = defaultdict(list)
    for r in csv2:
        groups2[(r["density"], r["scope"], r["scope_id"])].append(r)
        check(r["label"] == "exploratory, post hoc", "label 2")
    check(len(groups2) == 3 * 13 + 4, "lost_gained scopes")
    for (dens, sc, sid), rows in groups2.items():
        Ks = (1, 2, 3) if dens == "all" else (int(dens),)
        acts = actors if sc == "pooled" else [a for a in actors if f"seed{seed[a]}" == sid] if sc == "seed" else [sid]
        L, G, RG, net = tally([(K, a) for K in Ks for a in acts])
        n = 1000 * len(Ks) * len(acts)
        for r in rows:
            cat, dr, c = r["category"], r["direction"], int(r["count"])
            want = {"LOST": lambda: L[cat], "GAINED": lambda: G[cat], "NET": lambda: net,
                    "LOST_REACHED_REGION": lambda: RG[int(cat[-1])]}[dr]()
            check(c == want, f"{dens} {sid} {dr} {cat}: {c} vs {want}")
            check(int(r["n"]) == n, "n2")
            close(r["share"], c / (L["total"] if r["share_of"] == "lost" else n), 1e-12, "share")
        check(G["from_infeasible"] == 0, "gained from infeasible is zero")
        check(L["total"] - G["total"] == net, f"accounting {dens} {sid}")
        if sc == "pooled" and dens != "all":
            close(net / n, verdict[dens]["b"]["goal_loss"], 1e-12, f"criterion (b) {dens}")

    csv3 = list(csv.DictReader(open(RP / "transitions_4x4.csv")))
    check(len(csv3) == 48, "4x4 rows")
    for K in (1, 2, 3):
        T4 = Counter()
        for a in actors:
            T4.update(zip(outc[(K, a, "off")].tolist(), outc[(K, a, "on")].tolist()))
        rows = [r for r in csv3 if int(r["density"]) == K]
        check(len(rows) == 16 and sum(int(r["count"]) for r in rows) == 9000, f"4x4 total {K}")
        for r in rows:
            check(int(r["count"]) == T4[(r["off_outcome"], r["on_outcome"])], f"4x4 {K} {r['off_outcome']}->{r['on_outcome']}")

    status = "PASS" if not fails else "FAIL"
    pooled = {(r["density"], r["direction"], r["category"]): r for r in csv2 if r["scope"] == "pooled"}
    lines = [
        "# Verification of the on-vs-B-match and lost/gained outputs",
        "",
        f"**Result: {status}**, {checks['n']:,} checks, {len(fails)} failed.",
        "",
        "`scripts/verify_bmatch_transitions.py` does not import `cbf_study`, `scripts/paper_bmatch_transitions.py`",
        "or any project statistics helper. It recomputes the outputs from the saved final-test data:",
        "",
        "- **Wilson limits and Newcombe methods 10 and 8:** its own implementation, with the normal quantile from",
        "  the Python standard library. The interval endpoints agree to 1e-12, and the 2×2 counts agree exactly.",
        "- **Outcomes:** taken from the NPZ `outcome` arrays in `results/final_test/steps/`, not from the cell CSVs.",
        "- **Formation error e(t):** recomputed from the NPZ poses, with slots rebuilt from the protocol geometry",
        "  (radius 0.30 m, angles π/2 + 2πr/3). It uses the common prefix L = min(T_on, T_bmatch), and estimates",
        "  agree to 1e-8 m.",
        f"- **Bootstrap:** its own implementation of the locked draw order (B = 20,000, seed 20261026). Agreement:",
        f"  **{boot_mode}** (max |difference| {boot_maxdiff:.3g} m).",
        "- **LOST, GAINED and NET:**",
        "  - recounted per density, seed, actor and K = 1–3 pooled;",
        "  - LOST − GAINED equals the off GOAL count minus the on GOAL count in every scope;",
        "  - (LOST − GAINED)/n equals the unrounded criterion-(b) estimate in `verdict.json` to 1e-12.",
        "- **4×4 transitions:** all 48 cells recounted.",
        "- **TIMEOUT classes:** `timeout_classes.csv` covers every filter-on TIMEOUT exactly once.",
        "- **Data:** no duplicate keys; 9 actors × 1,000 trials × 3 arms per density; layout hashes identical",
        "  across arms; seeds agree with `checkpoint_selection.json`.",
        "- **Constructed fixtures:** all-negative, all-positive, zero-discordance, one-direction, negative association",
        "  and arm reversal.",
        "",
        "**Limitation:** the reached-goal-region split uses the saved filter-on flags `reached_goal_region`. Only",
        "their aggregation is verified here, not the flags themselves.",
        "",
        "Pooled K = 1–3:",
        "",
        f"- LOST {pooled[('all', 'LOST', 'total')]['count']}, GAINED {pooled[('all', 'GAINED', 'total')]['count']},",
        f"  net {pooled[('all', 'NET', 'lost_minus_gained')]['count']}.",
        f"- LOST pairs whose filtered run never reached the goal region:",
        f"  {pooled[('all', 'LOST_REACHED_REGION', 'on_reached_goal_region_0')]['count']}.",
    ]
    if fails:
        lines += ["", "Failures:"] + [f"- {f}" for f in fails[:50]]
    (RP / "bmatch_transitions_verification.md").write_text("\n".join(lines) + "\n")
    print(f"{status}: {checks['n']} checks, {len(fails)} failed; bootstrap {boot_mode} (max diff {boot_maxdiff:.3g})")
    for f in fails[:20]:
        print("  FAIL", f)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
