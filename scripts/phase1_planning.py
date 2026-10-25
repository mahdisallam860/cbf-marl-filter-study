#!/usr/bin/env python3
"""Phase 1, item 4: compute estimate and McNemar sample-size scenarios for the PROPOSED filter-on
validation sweep and final test (docs/PREFILTER_VALIDATION_PROTOCOL_DRAFT_2026-10-25.md).

Compute. Measured inputs only:
  * filter-off cost per control period at batch 200, per density, from results/phase1/eval/runs.csv
    (wall time / periods the batch ran; 4 processes on 4 cores);
  * a TIMING MICROBENCHMARK of one CBFQPFilter call on synthetic random states (numpy RNG, no layout
    sampler, no policy, no environment, no trial and no outcome). It measures solver cost only. Two
    cases: robots and obstacles spread over the field, and obstacles packed within 0.6 m of the robots
    (every row active, worst case).
  A filter-on run is bounded by max_steps periods (one long trial keeps the batch stepping), so its
  cost is estimated as max_steps x (filter-off period cost + filter call). The brake comparator is
  not implemented; it needs the same barrier rows but no QP, so the filter cost is used as its upper
  bound.

Sample size. The discordant-pair probabilities are ASSUMPTIONS listed on a grid, for both directions
of change, not estimates: no filter-on data exist. The only filter-off fact used is the bound
p10 <= P(event with the filter off), shown next to the grid. Connor (1987) closed form and the exact
McNemar test simulated (cbf_study.stats).

Writes results/phase1/planning/{filter_benchmark.csv, compute_estimate.json, sample_size_scenarios.csv,
PLANNING.md}."""
import csv, json, math, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "phase1" / "planning"

N_POLICIES = 9
PRIMARY_K = (1, 2, 3)
STRESS_K = (5, 8)
VAL_N = 200                                   # validation layouts per density (existing sets)
BRAKE_RUNS_PER_CELL = 1                       # brake comparator at the locked gain (proposal)
P10_GRID = (0.02, 0.05, 0.10, 0.20, 0.30)
P01_GRID = (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20)
GAIN_GRID = (0.02, 0.05, 0.10, 0.20)
LOSS_GRID = (0.0, 0.02, 0.05, 0.10, 0.20)
ALPHAS = (("two-sided 0.05", 0.05), ("Bonferroni 0.05/3 (three densities)", 0.05 / 3))
POWER = 0.8


def benchmark(cfg, reps=30):
    from cbf_study.cbf_qp import CBFQPFilter
    rng = np.random.default_rng(0)
    fc = cfg.filter_config(2.0)
    filt = CBFQPFilter(fc)
    slots = np.array([[0.0, 0.3], [-0.26, -0.15], [0.26, -0.15]])
    B, N = VAL_N, 3
    rows = []
    for K in (1, 2, 3, 4, 5, 8):
        for case in ("field", "packed"):
            cen = np.stack([rng.uniform(-1.2, 1.2, B), rng.uniform(-0.6, 0.6, B)], axis=1)
            poses = np.concatenate([cen[:, None] + slots[None], rng.uniform(-np.pi, np.pi, (B, N, 1))], axis=2)
            if case == "field":
                obst = np.stack([rng.uniform(-0.8, 0.8, (B, K)), rng.uniform(-0.9, 0.9, (B, K))], axis=2)
            else:
                ang, rad = rng.uniform(0, 2 * np.pi, (B, K)), rng.uniform(0.45, 0.6, (B, K))
                obst = cen[:, None] + rad[..., None] * np.stack([np.cos(ang), np.sin(ang)], axis=2)
            u = np.stack([rng.uniform(fc.v_min, fc.v_max, (B, N)), rng.uniform(-fc.w_max, fc.w_max, (B, N))], axis=2)
            filt(poses, u, obst)                                    # warm-up
            t0 = time.perf_counter()
            for _ in range(reps):
                filt(poses, u, obst)
            rows.append(dict(density=K, case=case, batch=B, reps=reps, s_per_call=(time.perf_counter() - t0) / reps))
    return rows


def compute(cfg, bench):
    runs = list(csv.DictReader(open(ROOT / "results/phase1/eval/runs.csv")))
    T = cfg.protocol_config().max_steps
    per = {}
    for K in sorted({int(r["density"]) for r in runs}):
        x = [float(r["wall_s"]) / int(r["batch_periods"]) for r in runs if int(r["density"]) == K]
        per[K] = float(np.median(x))
    fil = {(b["density"], b["case"]): b["s_per_call"] for b in bench}
    run_s = lambda K, case: T * (per[K] + fil[(K, case)])     # one filter-on run, 200 trials, upper bound
    off_s = lambda K: T * per[K]
    gains = cfg.alphas
    est = dict(max_steps=T, batch=VAL_N, filter_off_s_per_period=per, filter_call_s=fil, gains=gains,
               n_policies=N_POLICIES)

    def plan(densities, n_layouts, n_gains, brake):
        scale = n_layouts / VAL_N                              # cost ~ linear in batch for batches >= 200 (assumed)
        out = {}
        for case in ("field", "packed"):
            on = sum(run_s(K, case) for K in densities) * N_POLICIES * n_gains * scale
            br = sum(run_s(K, case) for K in densities) * N_POLICIES * brake * scale
            off = sum(off_s(K) for K in densities) * N_POLICIES * scale
            tot = on + br + off
            out[case] = dict(filter_on_core_h=on / 3600, brake_core_h=br / 3600, filter_off_core_h=off / 3600,
                             total_core_h=tot / 3600, wall_h_4_cores=tot / 3600 / 4)
        return out

    est["validation_sweep_primary"] = dict(
        what=f"{N_POLICIES} policies x K {PRIMARY_K} x {len(gains)} gains x {VAL_N} validation layouts, filter on; "
             f"brake comparator at each gain; filter off once", runs_filter_on=N_POLICIES * len(PRIMARY_K) * len(gains),
        **plan(PRIMARY_K, VAL_N, len(gains), len(gains)))
    est["validation_sweep_stress"] = dict(
        what=f"same at K {STRESS_K} (baseline stress, optional)", runs_filter_on=N_POLICIES * len(STRESS_K) * len(gains),
        **plan(STRESS_K, VAL_N, len(gains), len(gains)))
    est["final_test_per_1000_layouts_per_policy"] = dict(
        what=f"{N_POLICIES} policies x K {PRIMARY_K} x 1000 test layouts per policy and density: filter off, "
             f"filter on at the locked gain, brake at the locked gain", **plan(PRIMARY_K, 1000, 1, BRAKE_RUNS_PER_CELL))
    return est


def sample_sizes():
    from cbf_study.stats import mcnemar_n_connor, mcnemar_power_sim, min_n_for_power
    rows = []
    grids = [("COLLISION", "p10 = P(off collides, on does not)", "p01 = P(off does not collide, on collides)",
              P10_GRID, [p for p in P01_GRID if p <= 0.05]),
             ("COLLISION or INFEASIBLE", "p10 = P(off collides, on neither collides nor is infeasible)",
              "p01 = P(off collision-free, on collides or is infeasible)", P10_GRID, P01_GRID),
             ("GOAL", "p_gain = P(off not GOAL, on GOAL)", "p_loss = P(off GOAL, on not GOAL)", GAIN_GRID, LOSS_GRID)]
    for ep, d10, d01, g10, g01 in grids:
        for p10 in g10:
            for p01 in g01:
                for lab, a in ALPHAS:
                    nc = mcnemar_n_connor(p10, p01, alpha=a, power=POWER)
                    ns = (min_n_for_power(lambda n: mcnemar_power_sim(p10, p01, n, alpha=a, reps=4000, seed=21),
                                          n_start=max(10, nc // 2), n_max=200000) if nc <= 200000 else float("inf"))
                    rows.append(dict(endpoint=ep, p_a=p10, p_b=p01, def_a=d10, def_b=d01, net_effect=p01 - p10,
                                     alpha_label=lab, alpha=a, power=POWER, n_pairs_connor=nc, n_pairs_exact_sim=ns,
                                     n_per_policy_9_disjoint=(math.ceil(ns / N_POLICIES) if ns != float("inf") else float("inf"))))
    return rows


def offsite_bounds():
    """Filter-off event rates per density (min / max over the nine policies): the bound p10 <= P_off."""
    rows = list(csv.DictReader(open(ROOT / "results/phase1/tables/filter_off_by_policy_density.csv")))
    out = {}
    for K in (1, 2, 3, 4, 5, 8):
        rs = [r for r in rows if int(r["density"]) == K]
        coll = [(int(r["COLL_OBST"]) + int(r["COLL_ROBOT"])) / int(r["n"]) for r in rs]
        goal = [int(r["GOAL"]) / int(r["n"]) for r in rs]
        out[K] = dict(coll_min=min(coll), coll_max=max(coll), goal_min=min(goal), goal_max=max(goal))
    return out


def write_md(est, ss, bounds, bench):
    f = lambda x: "inf" if x == float("inf") else f"{int(x):,}"
    L = ["# Phase 1: compute estimate and McNemar sample-size scenarios (planning only)", "",
         "Generated by `scripts/phase1_planning.py`. **No filter-on or brake trial was run.** Filter-on cost comes from "
         "the measured filter-off cost per period plus a timing microbenchmark of one `CBFQPFilter` call on synthetic "
         "random states (no layouts, policy, environment or outcomes). The brake comparator is not implemented; "
         "the filter cost is used as its upper bound.", "",
         "## Measured costs (batch of 200 trials)", "",
         "| K | filter-off s per period (median over 9 runs, 4 parallel processes) | filter call s, field / packed | filter-on run upper bound (400 periods) s, field / packed |",
         "|---|---|---|---|"]
    for K in (1, 2, 3, 4, 5, 8):
        p = est["filter_off_s_per_period"][K]
        a, b = est["filter_call_s"][(K, "field")], est["filter_call_s"][(K, "packed")]
        L.append(f"| {K} | {p:.4f} | {a:.4f} / {b:.4f} | {400 * (p + a):.1f} / {400 * (p + b):.1f} |")
    L += ["", "## Compute estimates (core-hours; wall on 4 cores)", "",
          "| plan | filter-on runs | filter on | brake | filter off | total core-h | wall h on 4 cores |", "|---|---|---|---|---|---|---|"]
    for key in ("validation_sweep_primary", "validation_sweep_stress", "final_test_per_1000_layouts_per_policy"):
        e = est[key]
        for case in ("field", "packed"):
            c = e[case]
            L.append(f"| {e['what']} ({case} filter cost) | {e.get('runs_filter_on', '-')} | {c['filter_on_core_h']:.2f} | "
                     f"{c['brake_core_h']:.2f} | {c['filter_off_core_h']:.2f} | {c['total_core_h']:.2f} | {c['wall_h_4_cores']:.2f} |")
    L += ["", "Assumptions: every filter-on or brake run lasts the full 400 periods (upper bound; filter-off batches "
          "at K = 8 ended after ~30 periods because every trial collided, a filter-on batch may not); cost linear in "
          "batch size; 4 cores as in this container. The final-test row scales linearly with the layouts per policy "
          "chosen from the sample-size table below.", "",
          "## McNemar sample sizes: scenarios with explicit assumptions", "",
          f"Two-sided exact McNemar test, power {POWER}; n = pairs per test (closed form, Connor 1987 / exact test "
          "simulated, 4,000 replicates). Both directions of change are assumptions, not estimates. "
          "'inf' = no net effect (p10 = p01): no number of pairs gives power. If the nine policies each get a "
          "disjoint layout set and their pairs are pooled in one test per density (protocol draft, section 6), the "
          "layouts per policy are n / 9 (column `n_per_policy_9_disjoint` in `sample_size_scenarios.csv`).", "",
          "Filter-off facts that bound the scenarios (min-max over the nine policies, validation): "
          + "; ".join(f"K = {K}: collision {100 * b['coll_min']:.1f}-{100 * b['coll_max']:.1f}%, GOAL {100 * b['goal_min']:.1f}-{100 * b['goal_max']:.1f}%"
                      for K, b in bounds.items() if K in PRIMARY_K)
          + ". A scenario needs p10 <= P(collision, filter off) and p_gain <= 1 - P(GOAL, filter off); "
          "these bounds are the only use of filter-off data here.", ""]
    for ep in ("COLLISION", "COLLISION or INFEASIBLE", "GOAL"):
        rs = [r for r in ss if r["endpoint"] == ep]
        a_vals = sorted({r["p_a"] for r in rs})
        b_vals = sorted({r["p_b"] for r in rs})
        L += [f"### Endpoint {ep}", "", f"{rs[0]['def_a']}; {rs[0]['def_b']}.", "",
              "| " + ("p10" if ep != "GOAL" else "p_gain") + " | " + " | ".join(
                  f"{'p01' if ep != 'GOAL' else 'p_loss'} = {b:g}" for b in b_vals) + " |",
              "|---|" + "---|" * len(b_vals)]
        for a in a_vals:
            cells = []
            for b in b_vals:
                x = {r["alpha_label"]: r for r in rs if r["p_a"] == a and r["p_b"] == b}
                r1, r2 = x[ALPHAS[0][0]], x[ALPHAS[1][0]]
                cells.append(f"{f(r1['n_pairs_connor'])} / {f(r1['n_pairs_exact_sim'])} ({f(r2['n_pairs_exact_sim'])})")
            L.append(f"| {a:g} | " + " | ".join(cells) + " |")
        L += ["", "Cell: closed form / exact simulation at two-sided 0.05 (exact simulation at 0.05/3 in brackets).", ""]
    (OUT / "PLANNING.md").write_text("\n".join(L) + "\n")


def main():
    from cbf_study.config import StudyConfig
    from cbf_study.evaluate import write_rows
    import torch
    torch.set_num_threads(1)
    cfg = StudyConfig.load()
    OUT.mkdir(parents=True, exist_ok=True)
    bench = benchmark(cfg)
    write_rows(OUT / "filter_benchmark.csv", bench)
    est = compute(cfg, bench)
    ss = sample_sizes()
    write_rows(OUT / "sample_size_scenarios.csv", ss)
    bounds = offsite_bounds()
    js = dict(est, filter_call_s={f"K{k[0]}_{k[1]}": v for k, v in est["filter_call_s"].items()},
              filter_off_bounds=bounds)
    (OUT / "compute_estimate.json").write_text(json.dumps(js, indent=1, default=str))
    write_md(est, ss, bounds, bench)
    print((OUT / "PLANNING.md").read_text())


if __name__ == "__main__":
    main()
