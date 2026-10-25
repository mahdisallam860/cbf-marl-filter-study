# Phase 1 (2026-10-25): filter-off validation, timeout diagnosis, planning. FILTER-OFF ONLY

No training was run, and no filter-on or brake trial. No final-test layout was generated or
inspected. Tolerances, velocity limits, the filter and the scoring code are unchanged. Every number
below is a filter-off result on the validation layouts, which had already been used (disclosure:
`docs/PREFILTER_VALIDATION_PROTOCOL_DRAFT_2026-10-25.md` §0). All config values are still pending
advisor approval.

## Files

| item | files |
|---|---|
| 1 checkpoint rule and IDs (committed `d598055`, before any evaluation) | `checkpoint_selection.{md,json}`, `scripts/phase1_select_checkpoints.py` |
| 2 solvability, K = 1, 3, 4 | `solvability/`, `scripts/phase1_solvability.py` |
| 2 filter-off tables | `TABLES.md`, `tables/filter_off_by_policy_density.csv`, `tables/gate_v2_consistency.csv`, `eval/{trials,runs}.csv`, `scripts/phase1_eval.py`, `scripts/phase1_tables.py` |
| 3 timeout diagnosis | `TIMEOUT_DIAGNOSIS.md`, `tables/timeout_diagnosis*.csv` |
| 3 exploratory tolerance sensitivity | `TOLERANCE_SENSITIVITY.md`, `tables/tolerance_sensitivity.csv` |
| 4 compute and sample size | `planning/{PLANNING.md, compute_estimate.json, filter_benchmark.csv, sample_size_scenarios.csv}`, `scripts/phase1_planning.py` |
| 5 protocol draft and seed reservation | `docs/PREFILTER_VALIDATION_PROTOCOL_DRAFT_2026-10-25.md`, `docs/final_test_seed_reservation.json`, `tests/test_seed_reservation.py` |

## 1. Checkpoints

The rule uses the final stage only, with n = 326 final-stage iterations: early = first saved
checkpoint with r ≥ ⌈n/3⌉, middle = first with r ≥ ⌈2n/3⌉, late = r = n. It gives r = 120, 220
and 326 for every seed:

* Seed 0: iterations 526, 626, 732.
* Seeds 1 and 2: iterations 506, 606, 712.

The rule picks none of the per-seed checkpoints that the gate_v2 rule had picked (506, 666, 586).
Seed 1's middle and late checkpoints come from the resumed run (audit §3).

## 2. Filter-off tables

The validation layouts at K = 1, 3 and 4 are all solvable: 200/200 FOUND in (a), (b) and (c).
Ranges over the nine policies, 200 trials per cell, with Wilson CIs per cell in `TABLES.md`:

| K | GOAL | obstacle collision | robot collision | TIMEOUT (all stalled but one) | formation error, mean [m] |
|---|---|---|---|---|---|
| 1 | 17.0–82.0 % | 3.0–11.5 % | 0 | 13.0–72.5 % | 0.031–0.037 |
| 2 | 48.5–79.5 % | 10.5–26.0 % | 0–1.0 % | 5.0–28.0 % | 0.035–0.046 |
| 3 | 14.0–58.0 % | 27.0–63.0 % | 0–3.5 % | 14.5–28.0 % | 0.036–0.050 |
| 4 (diagnostic) | 5.0–19.5 % | 69.0–83.5 % | 0–1.0 % | 4.5–15.0 % | 0.036–0.057 |
| 5 (stress) | 0–4.0 % | 89.5–98.5 % | 0–0.5 % | 1.5–6.0 % | 0.040–0.054 |
| 8 (stress) | 0 % | 100 % | 0 | 0 % | 0.040–0.055 |

INFEASIBLE does not occur with the filter off. At K = 2, 5 and 8 every cell matches the gate_v2
counts exactly (27/27), and a second complete run reproduced all 10,800 trials.

Policy order differs by density. For example, s0_early has the highest GOAL at K = 1 (82 %) but
one of the lowest at K = 3 (15.5 %), while s2_early has the highest at K = 3 (58 %). Collisions are
almost all robot–obstacle.

## 3. Near-goal timeouts (tolerances unchanged: 0.10 m and 0.10 m)

What the data show (`TIMEOUT_DIAGNOSIS.md`), with the nine policies pooled:

* **Almost every timeout ends near the goal.** The share of timeouts with a final centroid
  distance ≤ 0.30 m is 775/775 at K = 1, 308/312 at K = 2 and 361/381 at K = 3. The median final
  distance is 0.15–0.16 m (q10–q90 about 0.11–0.24 m).
* **The unmet criterion is the centroid distance.** In 766 + 308 + 358 of these trials the
  centroid criterion alone fails. The median final mean slot error is 0.03 m, well inside the
  formation tolerance.
* **The team stops.**
  * 86 % of the near-goal timeouts stop completely: every applied command stays below 0.01 from
    some period on. The median stop starts at period 122–156, after which the team stands still
    for about 240–280 periods.
  * In the last 50 periods the policy's own raw action is about zero (q90: |v| ≤ 0.016 m/s,
    |ω| ≤ 0.10 rad/s).
  * The deterministic closed loop therefore has resting points 0.11–0.24 m from the goal.
* **The remaining error is mostly sideways to the robots.** At the end of near-goal timeouts the
  error to the target is 0.13–0.14 m lateral and 0.04–0.06 m longitudinal (median, body frame).
  GOAL trials end at 0.05 m lateral and 0.08 m longitudinal. A unicycle can remove a lateral offset
  only by turning first.
* **No obstacle is involved.** At the end, no robot is within the 0.15 m proximity-penalty margin
  of an obstacle in 0 of 775 (K = 1), 0 of 308 (K = 2) and 8 of 361 (K = 3) near-goal timeouts. The
  nearest obstacle is as far from the goal as in GOAL trials (K = 2: 1.36 m vs 1.34 m median), or
  farther (K = 1: 1.78 m vs 1.53 m).
* **K = 1 is the worst case.** It has the most near-goal stalls (TIMEOUT up to 72.5 %). K = 1 was
  last trained in curriculum stage 1; the final stage used K = 2, 5 and 8 only.

**Interpretation, not tested here.** These are terminal-precision failures of the learned
deterministic policy, not obstacle-avoidance failures. Plausible contributors:

* the reward has no goal bonus and no time penalty, so a residual of 0.15 m is worth only 1.5 in
  progress reward;
* correcting a lateral offset needs a turn-in-place manoeuvre;
* evaluation uses the action mean (deterministic), while training used a stochastic policy.

**Exploratory sensitivity (`TOLERANCE_SENSITIVITY.md`; exact for looser tolerances).** Pooled GOAL
counts under a looser goal tolerance (formation tolerance kept at 0.10 m):

| K | goal tolerance 0.10 m | 0.15 m | 0.20 m |
|---|---|---|---|
| 1 | 51.1 % | 76.2 % | 88.3 % |
| 2 | 65.1 % | 76.1 % | 79.9 % |
| 3 | 31.9 % | 43.6 % | 49.1 % |

Loosening the formation tolerance changes almost nothing. At K = 1, 0.10 → 0.15 m adds 2 of 1,800
trials, and none at K = 2 or 3.

**No change to the tolerances is proposed.**

* Physical: none found.
  * The 0.10 m goal tolerance equals the robot radius and is twice the largest distance a robot
    travels in one period (v_max·Δ = 0.05 m).
  * The state is exact (A2).
  * The geometric audit reaches the goal in every layout.
  * 51–65 % of the trials at K = 1 and 2 meet the tolerance, ending with a lateral error of about
    0.05 m.
* Task-based: one consideration only, left to the advisors. The stalls happen away from obstacles,
  so the GOAL endpoint mixes parking precision with obstacle-related failures; the COLLISION
  endpoint is unaffected. If the advisors want a completion measure insensitive to parking
  precision, it would have to be pre-registered as a separately named secondary endpoint before
  any filter-on run. GOAL stays as defined.

## 4. Compute and sample size (`planning/PLANNING.md`)

**Measured inputs:**

* Filter-off cost: 0.011–0.034 s per period at batch 200 (K = 1–8).
* One filter call: 0.005–0.027 s, from a timing microbenchmark on synthetic random states. That
  is not a trial: no layout, policy or outcome is involved.

**Upper bounds**, with every run lasting 400 periods and the brake costing no more than the filter:

| plan | core-hours | wall on 4 cores |
|---|---|---|
| validation sweep: nine policies × K = 1–3 × five gains, filter on and brake | about 0.7 | about 11 min |
| optional stress cells, K = 5 and 8 | about 1.0 | about 15 min |
| final test: nine policies × K = 1–3 × 1,000 layouts each, off, on at α\* and brake | about 0.9 | about 13 min |

**McNemar scenarios.** The grids cover both directions of change. They are assumptions, not
estimates; the only filter-off fact used is the bound p10 ≤ P_off(event).

* **COLLISION:** p10 from 0.02 to 0.30, p01 from 0 to 0.05.
* **COLLISION or INFEASIBLE:** p01 up to 0.20, because the filter can turn a collision-free trial
  into an INFEASIBLE one.
* **GOAL:** p_gain from 0.02 to 0.20, p_loss from 0 to 0.20.

Examples for COLLISION, exact test, power 0.8, 0.05/3:

| p10 | p01 | pairs |
|---|---|---|
| 0.05 | 0 | 180 |
| 0.05 | 0.01 | 402 |
| 0.02 | 0.005 | 1,196 |
| 0.02 | 0.01 | 3,285 |

## 5. Protocol draft

`docs/PREFILTER_VALIDATION_PROTOCOL_DRAFT_2026-10-25.md` lists:

* the candidate densities: 1, 2 and 3; 5 and 8 as stress; 4 as a diagnostic;
* the nine fixed checkpoints;
* the metrics, with COLLISION proposed as the primary endpoint;
* the gain grid {0.5, 1, 2, 5, 10} with its selection rule: the highest pooled filter-on
  validation GOAL wins, ties go to the fewest INFEASIBLE, then to the larger α;
* the proposed brake comparator (B-match). It is not implemented;
* the final-test design and the lock sequence;
* the disclosure of every filter-off result already seen.

Final-test seeds are reserved in 4,000,000–4,999,999 (4,000,000 + 1,000·p + K) and have not been
generated. A test checks that they are disjoint from every used seed and absent from all saved
results. Advisor agreement on the limits has not been obtained; the draft records this.
