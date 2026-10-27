#!/usr/bin/env python3
"""Two saved-data analyses of the final test (no simulation, no new layouts, no new verdicts).

1. Filter on vs B-match at alpha* (PRE-DECLARED SECONDARY, DESCRIPTIVE; locked protocol, secondary 6).
   Paired differences, always on minus bmatch, for GOAL, failure (COLLISION or INFEASIBLE) and TIMEOUT
   on all 9,000 pairs per density, with two-sided 95% Newcombe (1998) method-10 intervals (Wilson
   score limits, continuity-corrected phi). Method 8 (uncorrected phi) is written to the CSV only.
   2x2 counts: a = event in both arms, b = on only, c = bmatch only, d = neither; estimate (b - c)/n.
   phi: numerator ad - bc, replaced by max(ad - bc - n/2, 0) when ad - bc > 0 (method 10 only); if any
   marginal total is zero (zero marginal variance, e.g. failure: B-match had no failures) the
   correlation is undefined and phi = 0 is used. The interval is always computed; zero discordance
   does not give [0, 0]. Formation: d_i = mean_{t=1..L_i} [e_on(t) - e_bmatch(t)], L_i = min(T_on,
   T_bmatch), e(t) recomputed from the saved post-step poses; L_i = 0 is excluded and counted.
   Equal-actor mean; 95% stratified paired percentile bootstrap within actor (B = 20,000,
   default_rng(20261026), 2.5% / 97.5% quantiles, method 'linear'), pooled rows only. This is a new
   code path; the historical criterion-(c) code is neither reused nor changed.
2. Lost and gained completions, filter off -> filter on (EXPLORATORY, POST HOC). LOST = off GOAL and on
   not GOAL; GAINED = off not GOAL and on GOAL. TIMEOUT categories are read from
   results/final_test/mechanism/timeout_classes.csv. 4x4 off x on transition tables per density.
   Reached-goal-region split of the LOST pairs uses the saved filter-on flag (0 / 1).

Reads results/final_test/{cells,steps,layouts}/, mechanism/timeout_classes.csv, verdict.json,
results/phase1/checkpoint_selection.json and the committed paper/numbers.tex (TIMEOUT-class macros).
Writes results/paper_revision/{on_vs_bmatch.csv, lost_gained.csv, transitions_4x4.csv}.
"""
import csv, gzip, json, re, subprocess, sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbf_study.stats import wilson_ci   # noqa: E402

FT = ROOT / "results" / "final_test"
OUT = ROOT / "results" / "paper_revision"
KS = (1, 2, 3)
OUTCOMES = {1: "GOAL", 2: "COLLISION", 3: "INFEASIBLE", 4: "TIMEOUT"}
ORDER = ("GOAL", "COLLISION", "INFEASIBLE", "TIMEOUT")
ARMS = ("off", "on", "bmatch")
CONF = 0.95
REPS, SEED = 20000, 20261026
LAB1 = "pre-declared secondary, descriptive"
def R(x):
    return repr(float(x))


LAB2 = "exploratory, post hoc"
FI = ("filter-induced stall: obstacle", "filter-induced stall: neighbour", "filter-induced stall: obstacle+neighbour")
OTHER_TO = ("near-goal parking stall", "other stall", "slow progress")


# ---------------------------------------------------------------- paired interval
def newcombe(a, b, c, d, conf=CONF, corrected=True):
    """p_on - p_bmatch with a both, b on only, c bmatch only, d neither. corrected=True: method 10."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson_ci(a + b, n, conf)
    l2, u2 = wilson_ci(a + c, n, conf)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    nu = float(a * d - b * c)
    if corrected and nu > 0:
        nu = max(nu - n / 2, 0.0)
    phi = nu / np.sqrt(float(den)) if den > 0 else 0.0
    th = p1 - p2
    dl = np.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    du = np.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return th, max(-1.0, th - dl), min(1.0, th + du)


def constructed_checks():
    """Fixtures for the paired interval (both methods). Returns the number of checks."""
    n_checks = 0
    cases = {"all negative": (0, 0, 0, 100), "all positive": (100, 0, 0, 0), "identical mixed, no discordance": (30, 0, 0, 70),
             "one-direction discordance": (20, 15, 0, 65), "negative association": (0, 40, 35, 25), "large": (3000, 40, 7, 5953),
             "zero marginal (bmatch none)": (0, 9, 0, 8991)}
    for name, (a, b, c, d) in cases.items():
        for corr in (True, False):
            th, lo, hi = newcombe(a, b, c, d, corrected=corr)
            assert np.isfinite([th, lo, hi]).all() and -1 <= lo <= th <= hi <= 1, (name, corr, th, lo, hi)
            assert abs(th - (b - c) / (a + b + c + d)) < 1e-15, name
            n_checks += 2
            r, rlo, rhi = newcombe(a, c, b, d, corrected=corr)        # arm reversal
            assert abs(r + th) < 1e-12 and abs(rlo + hi) < 1e-12 and abs(rhi + lo) < 1e-12, (name, "reversal")
            n_checks += 1
            if b == c == 0:
                assert hi - lo > 0, (name, "zero discordance must not give [0, 0]")
                n_checks += 1
    for case in ((0, 0, 0, 100), (100, 0, 0, 0)):          # all negative / all positive: symmetric about 0
        th, lo, hi = newcombe(*case)
        assert th == 0 and lo < 0 < hi and abs(lo + hi) < 1e-12, case
    th, lo, hi = newcombe(20, 15, 0, 65)
    assert lo > 0, "one-direction discordance of 15/100 excludes 0"
    w10 = np.subtract(*newcombe(30, 0, 0, 70)[2:0:-1]); w8 = np.subtract(*newcombe(30, 0, 0, 70, corrected=False)[2:0:-1])
    assert w10 >= w8, "continuity correction does not narrow a positively associated interval"
    n_checks += 4
    return n_checks


# ---------------------------------------------------------------- data
def formation_error(poses, slots):
    xy = poses[..., :2].astype(np.float64)
    rel = xy - xy.mean(axis=-2, keepdims=True)
    return np.linalg.norm(rel - slots, axis=-1).mean(axis=-1)


def load():
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())["policies"]
    actors = [p["policy_id"] for p in sel]
    seed_of = {p["policy_id"]: int(p["seed"]) for p in sel}
    assert len(actors) == 9
    D = {}
    for K in KS:
        for pid in actors:
            rows = {arm: {} for arm in ARMS}
            for r in csv.DictReader(gzip.open(FT / "cells" / f"{pid}_K{K}.csv.gz", "rt")):
                assert r["policy_id"] == pid and int(r["density"]) == K and int(r["seed"]) == seed_of[pid]
                t = int(r["trial"])
                assert t not in rows[r["condition"]], ("duplicate", pid, K, r["condition"], t)
                rows[r["condition"]][t] = r
            for arm in ARMS:
                assert sorted(rows[arm]) == list(range(1000)), (pid, K, arm)
            for t in range(1000):
                assert rows["off"][t]["layout_hash"] == rows["on"][t]["layout_hash"] == rows["bmatch"][t]["layout_hash"]
                assert rows["off"][t]["layout_seed"] == rows["on"][t]["layout_seed"] == rows["bmatch"][t]["layout_seed"]
            slots = np.load(FT / "layouts" / f"{pid}_K{K}.npz")["slots"].astype(np.float64)
            cell = {}
            for arm in ARMS:
                z = dict(np.load(FT / "steps" / f"{pid}_K{K}_{arm}.npz"))
                assert str(z["condition"]) == arm and z["outcome"].shape == (1000,)
                out = np.array([OUTCOMES[int(o)] for o in z["outcome"]])
                assert all(out[t] == rows[arm][t]["outcome"] for t in range(1000)), (pid, K, arm, "outcome")
                T = z["executed_periods"].astype(int)
                assert all(T[t] == int(rows[arm][t]["executed_periods"]) for t in range(1000))
                off = z["pose_offsets"]
                assert (np.diff(off) == T + 1).all()
                e = formation_error(z["poses"], slots) if arm != "off" else None
                cell[arm] = dict(out=out, T=T, off=off, e=e,
                                 rgr=np.array([int(rows[arm][t]["reached_goal_region"]) for t in range(1000)]))
            cell["saved_d_on_bm"] = np.array([float(rows["bmatch"][t]["d_on_minus_bmatch"]) for t in range(1000)])
            D[(K, pid)] = cell
    return actors, seed_of, D


def pair_formation(cell):
    on, bm = cell["on"], cell["bmatch"]
    d = np.full(1000, np.nan)
    for i in range(1000):
        L = min(on["T"][i], bm["T"][i])
        if L == 0:
            continue
        eo = on["e"][on["off"][i] + 1: on["off"][i] + 1 + L]
        eb = bm["e"][bm["off"][i] + 1: bm["off"][i] + 1 + L]
        d[i] = float(np.mean(eo - eb))
    return d


def bootstrap(groups):
    """Stratified paired percentile bootstrap within actor, equal-actor weights (locked procedure)."""
    rng = np.random.default_rng(SEED)
    P = len(groups)
    boot = np.zeros(REPS)
    for g in groups:
        done = 0
        while done < REPS:
            m = min(REPS - done, max(1, 2_000_000 // len(g)))
            boot[done:done + m] += g[rng.integers(0, len(g), (m, len(g)))].mean(axis=1) / P
            done += m
    return [float(x) for x in np.quantile(boot, [0.025, 0.975], method="linear")]


def committed_macros():
    txt = subprocess.run(["git", "show", "HEAD:paper/numbers.tex"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return {m.group(1): m.group(2) for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{(.*)\}", txt)}


def main():
    n_checks = constructed_checks()
    actors, seed_of, D = load()
    scopes = lambda K: [("pooled", "all", actors)] + [("seed", f"seed{s}", [a for a in actors if seed_of[a] == s]) for s in (0, 1, 2)] + \
                       [("actor", a, [a]) for a in actors]

    # ---------------- analysis 1
    rows1 = []
    form_d = {(K, a): pair_formation(D[(K, a)]) for K in KS for a in actors}
    max_saved_diff = max(np.nanmax(np.abs(form_d[k] - D[k]["saved_d_on_bm"])) for k in form_d)
    for k in form_d:
        assert np.array_equal(np.isnan(form_d[k]), np.isnan(D[k]["saved_d_on_bm"])), "eligibility differs from saved column"
    for K in KS:
        for scope, sid, acts in scopes(K):
            on = np.concatenate([D[(K, a)]["on"]["out"] for a in acts])
            bm = np.concatenate([D[(K, a)]["bmatch"]["out"] for a in acts])
            n = len(on)
            for ep, f in (("GOAL", lambda o: o == "GOAL"), ("FAIL", lambda o: (o == "COLLISION") | (o == "INFEASIBLE")),
                          ("TIMEOUT", lambda o: o == "TIMEOUT")):
                x, y = f(on), f(bm)
                a, b, c, d = int((x & y).sum()), int((x & ~y).sum()), int((~x & y).sum()), int((~x & ~y).sum())
                th, lo, hi = newcombe(a, b, c, d)
                _, lo8, hi8 = newcombe(a, b, c, d, corrected=False)
                assert abs(th - (b - c) / n) < 1e-15
                rows1.append(dict(label=LAB1, endpoint=ep, density=K, scope=scope, scope_id=sid, n=n, rate_on=R(x.mean()),
                                  rate_bmatch=R(y.mean()), a=a, b=b, c=c, d=d, estimate=R(th), m10_lo=R(lo), m10_hi=R(hi),
                                  m8_lo=R(lo8), m8_hi=R(hi8), eligible="", excluded="", boot_lo="", boot_hi=""))
            groups = [form_d[(K, a)][~np.isnan(form_d[(K, a)])] for a in acts]
            assert all(len(g) > 0 for g in groups)
            est = float(np.mean([g.mean() for g in groups]))
            blo, bhi = bootstrap(groups) if scope == "pooled" else ("", "")
            rows1.append(dict(label=LAB1, endpoint="FORMATION_M", density=K, scope=scope, scope_id=sid, n=n, rate_on="", rate_bmatch="",
                              a="", b="", c="", d="", estimate=R(est), m10_lo="", m10_hi="", m8_lo="", m8_hi="",
                              eligible=sum(len(g) for g in groups), excluded=n - sum(len(g) for g in groups),
                              boot_lo=R(blo) if blo != "" else "", boot_hi=R(bhi) if bhi != "" else ""))
    for r in rows1:
        if r["endpoint"] != "FORMATION_M":
            assert np.isfinite([float(r["m10_lo"]), float(r["m10_hi"])]).all() and -1 <= float(r["m10_lo"]) <= float(r["estimate"]) <= float(r["m10_hi"]) <= 1
            n_checks += 1

    # ---------------- analysis 2
    tc = {}
    for r in csv.DictReader(open(FT / "mechanism/timeout_classes.csv")):
        if r["condition"] != "on":
            continue
        key = (r["policy_id"], int(r["density"]), int(r["trial"]))
        assert key not in tc, ("duplicate timeout class", key)
        tc[key] = r["category"]
    on_to = {(a, K, t) for K in KS for a in actors for t in np.flatnonzero(D[(K, a)]["on"]["out"] == "TIMEOUT").tolist()}
    assert set(tc) == on_to, "timeout_classes.csv does not cover the filter-on TIMEOUTs exactly once"
    assert set(tc.values()) <= set(FI + OTHER_TO), set(tc.values())
    cm = committed_macros()
    c_ = lambda x: f"{x:,}".replace(",", "{,}")
    assert cm["TOonTotal"] == c_(len(tc)) and cm["TOfi"] == c_(sum(v in FI for v in tc.values())) and \
        cm["TOng"] == c_(sum(v == "near-goal parking stall" for v in tc.values())), "timeout classes disagree with numbers.tex"
    n_checks += 4
    verdict = json.loads((FT / "verdict.json").read_text())["verdict_per_density"]

    rows2, rows3 = [], []
    cats_lost = ([("filter_induced_stall", lambda o, c: o == "TIMEOUT" and c in FI)] +
                 [(f"filter_induced_stall:{c.split(': ')[1]}", (lambda cc: lambda o, c: o == "TIMEOUT" and c == cc)(c)) for c in FI] +
                 [("other_timeout", lambda o, c: o == "TIMEOUT" and c in OTHER_TO)] +
                 [(f"other_timeout:{c.replace(' ', '_').replace('-', '_')}", (lambda cc: lambda o, c: o == "TIMEOUT" and c == cc)(c)) for c in OTHER_TO] +
                 [("infeasible", lambda o, c: o == "INFEASIBLE"), ("collision", lambda o, c: o == "COLLISION")])

    def lg_rows(Klab, scope, sid, keys):
        """keys: list of (K, actor) cells."""
        n = 1000 * len(keys)
        lost, gained, rgr = Counter(), Counter(), Counter()
        off_goal = on_goal = 0
        for (K, a) in keys:
            cell = D[(K, a)]
            off_o, on_o, rg = cell["off"]["out"], cell["on"]["out"], cell["on"]["rgr"]
            off_goal += int((off_o == "GOAL").sum()); on_goal += int((on_o == "GOAL").sum())
            for t in range(1000):
                if off_o[t] == "GOAL" and on_o[t] != "GOAL":
                    cat = tc.get((a, K, t), "")
                    for name, f in cats_lost:
                        lost[name] += f(on_o[t], cat)
                    lost["total"] += 1
                    rgr[int(rg[t])] += 1
                elif off_o[t] != "GOAL" and on_o[t] == "GOAL":
                    gained[f"from_{off_o[t].lower()}"] += 1
                    gained["total"] += 1
        assert gained["from_infeasible"] == 0
        assert lost["total"] == lost["filter_induced_stall"] + lost["other_timeout"] + lost["infeasible"] + lost["collision"]
        assert lost["filter_induced_stall"] == sum(lost[f"filter_induced_stall:{c.split(': ')[1]}"] for c in FI)
        assert lost["other_timeout"] == sum(lost[k] for k in lost if k.startswith("other_timeout:"))
        assert gained["total"] == gained["from_collision"] + gained["from_timeout"] + gained["from_infeasible"]
        assert lost["total"] - gained["total"] == off_goal - on_goal, (Klab, sid, "accounting")
        assert rgr[0] + rgr[1] == lost["total"]
        out = []
        add = lambda direction, cat, cnt, denom_label="all_pairs": out.append(dict(
            label=LAB2, density=Klab, scope=scope, scope_id=sid, n=n, direction=direction, category=cat, count=cnt,
            share=R(cnt / (lost["total"] if denom_label == "lost" else n)) if (lost["total"] or denom_label != "lost") else "", share_of=denom_label))
        for name, _ in cats_lost:
            add("LOST", name, lost[name])
        add("LOST", "total", lost["total"])
        for name in ("from_collision", "from_timeout", "from_infeasible", "total"):
            add("GAINED", name, gained[name])
        add("NET", "lost_minus_gained", lost["total"] - gained["total"])
        add("NET", "off_goal_minus_on_goal", off_goal - on_goal)
        add("LOST_REACHED_REGION", "on_reached_goal_region_0", rgr[0], "lost")
        add("LOST_REACHED_REGION", "on_reached_goal_region_1", rgr[1], "lost")
        return out, lost["total"] - gained["total"], n

    for K in KS:
        for scope, sid, acts in scopes(K):
            out, net, n = lg_rows(K, scope, sid, [(K, a) for a in acts])
            rows2 += out
            if scope == "pooled":
                assert abs(net / n - verdict[str(K)]["b"]["goal_loss"]) < 1e-12, (K, "criterion (b)")
                n_checks += 1
            n_checks += 1
        # 4x4 transitions
        T4 = Counter()
        for a in actors:
            for o1, o2 in zip(D[(K, a)]["off"]["out"], D[(K, a)]["on"]["out"]):
                T4[(o1, o2)] += 1
        for o1 in ORDER:
            assert sum(T4[(o1, o2)] for o2 in ORDER) == sum(int((D[(K, a)]["off"]["out"] == o1).sum()) for a in actors)
            assert sum(T4[(o2, o1)] for o2 in ORDER) == sum(int((D[(K, a)]["on"]["out"] == o1).sum()) for a in actors)
            n_checks += 2
            for o2 in ORDER:
                rows3.append(dict(label=LAB2, density=K, off_outcome=o1, on_outcome=o2, count=T4[(o1, o2)]))
        assert sum(T4.values()) == 9000
    for scope, sid, acts in [("pooled", "all", actors)] + [("seed", f"seed{s}", [a for a in actors if seed_of[a] == s]) for s in (0, 1, 2)]:
        out, _, _ = lg_rows("all", scope, sid, [(K, a) for K in KS for a in acts])
        rows2 += out

    OUT.mkdir(parents=True, exist_ok=True)
    order_scope = {"pooled": 0, "seed": 1, "actor": 2}
    akey = lambda s: actors.index(s) if s in actors else -1
    rows1.sort(key=lambda r: (r["density"], ("GOAL", "FAIL", "TIMEOUT", "FORMATION_M").index(r["endpoint"]), order_scope[r["scope"]], r["scope_id"] if r["scope"] == "seed" else "", akey(r["scope_id"])))
    rows2.sort(key=lambda r: (str(r["density"]), order_scope[r["scope"]], r["scope_id"] if r["scope"] == "seed" else "", akey(r["scope_id"])))
    for name, rows in (("on_vs_bmatch.csv", rows1), ("lost_gained.csv", rows2), ("transitions_4x4.csv", rows3)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
            w.writeheader(); w.writerows(rows)
    print(f"checks passed: {n_checks}; rows: {len(rows1)}, {len(rows2)}, {len(rows3)}; "
          f"max |d_on_bm (recomputed, float32 poses) - saved d_on_minus_bmatch| = {max_saved_diff:.3e}")


if __name__ == "__main__":
    main()
