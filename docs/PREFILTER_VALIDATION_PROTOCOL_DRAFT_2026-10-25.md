# Pre-filter validation protocol: DRAFT of 2026-10-25

**Status: draft, not locked, not approved.** Advisor agreement on the limits has **not** been
obtained. That covers the velocity bounds, radii, control period, goal and formation tolerances,
densities, barrier gains and gate thresholds: all 52 values in `config/study.toml` are still
`pending`. This draft proposes how the filter-on validation sweep and the final test would run once
those values and this procedure are agreed and locked. No filter-on or brake-comparator trial has
been run, and no final-test layout has been generated or inspected.

Safety statements stay within `docs/CBF_DESIGN.md`, Lemma 1 under assumptions A1–A9. Numerical
results from this protocol are not claims of continuous-time safety.

## 0. Disclosure: what has already been seen

Filter-OFF validation results were seen before this draft was written. Everything they informed
here (candidate densities, the checkpoint rule, the planning scenarios) was chosen with that
knowledge:

* `results/gate/`: config-v1 policies, every checkpoint, validation K = 2, 5, 8.
* `results/curriculum/*/heldout.csv`: config-v2 curriculum checkpoints, validation K = 0, 1, 2, 3
  (stage ends) and K = 2, 5, 8 (final stage).
* `results/gate_v2/`: all 51 final-stage checkpoints, validation K = 2, 5, 8.
* `results/phase1/`: the nine policies below at validation K = 1, 2, 3, 4, 5, 8, including the
  near-goal timeout diagnosis and the exploratory tolerance sensitivity.

Consequences:

* The validation layouts (seeds 2,000,000 + K) are not untouched. They may be used for selection
  (the gain sweep below), never for the final claim.
* The pilot layouts (seeds 3,000,000 + K) were inspected geometrically by the solvability and
  provenance audits, so they are not untouched either. They are not used for the final test.
* The final test uses only the reserved, never-generated seeds in section 7.

## 1. Fixed and unchanged

* The filter (`cbf_qp.py`, `qp2d.py`), the geometry and scoring (`geometry.py`, `protocol.py`)
  and the outcome precedence. INFEASIBLE is a reported outcome, never repaired.
* Goal tolerance 0.10 m (centroid to goal) and formation tolerance 0.10 m (mean slot error).
  The Phase 1 sensitivity analysis is exploratory only (`results/phase1/TOLERANCE_SENSITIVITY.md`).
* Velocity box v ∈ [−0.5, 0.5] m/s, |ω| ≤ 1 rad/s, Δ = 0.1 s, max_steps = 400.
* Config v2, env fingerprint `c60352aee3f11ed8`.
* Filter-on and filter-off runs are built from one config and differ only in α
  (`config.check_paired`).

## 2. Candidate densities (pending advisor decision)

| K | role | filter-off validation, nine policies (min–max) | reason |
|---|---|---|---|
| 1 | primary candidate | collision 3.0–11.5 %, GOAL 17.0–82.0 %, TIMEOUT 13.0–72.5 % | low-density end. Not in the final training stage; most failures are near-goal stalls |
| 2 | primary candidate | collision 10.5–26.0 %, GOAL 48.5–79.5 % | in the final training stage, and the best-learned density |
| 3 | primary candidate | collision 27.5–63.0 %, GOAL 14.0–58.0 % | high collision rate leaves room for an effect, and the policies still often reach the goal |
| 4 | intermediate diagnostic only | collision 70.0–83.5 %, GOAL 5.0–19.5 % | layouts solvable 200/200; not a study density |
| 5, 8 | baseline stress only | collision 90.0–100 %, GOAL 0–4 % | the policies drive into the obstacle field; at most a secondary stress analysis |

All validation layouts at K = 1, 2, 3, 4, 5, 8 are solvable by the geometry-only audit (200/200
each). Moving from the configured densities (2, 5, 8) to (1, 2, 3) changes the study question,
from dense fields to sparse ones, and needs advisor agreement.

## 3. Policy family (fixed; no further checkpoint selection)

There are nine policies: seeds 0, 1, 2 × early, middle and late checkpoints of the final
curriculum stage. They come from the iteration rule in `results/phase1/checkpoint_selection.md`,
which was committed in `d598055` before any Phase 1 evaluation:

* r = final-stage iteration.
* early = first saved checkpoint with r ≥ ⌈n/3⌉.
* middle = first saved checkpoint with r ≥ ⌈2n/3⌉.
* late = r = n, where n = 326.

The resulting checkpoints are iterations 526, 626 and 732 for seed 0 and 506, 606 and 712 for
seeds 1 and 2. File SHA-256 and actor digests are listed in that file, and every run checks the
actor digest.

No validation result selects or drops a policy. The family is the unit of analysis, so the result
is not tied to one hand-picked checkpoint.

The original gate (pooled GOAL ≥ 50 % over K = 2, 5, 8) failed for every checkpoint. Whether a
revised gate applies, and with which thresholds, is an advisor decision (D3). Thresholds proposed
now would be post hoc, because the Phase 1 numbers are already known.

## 4. Metrics

Each trial is logged as in `docs/CBF_DESIGN.md` §6, with the same code path for all conditions.

* **Primary (proposed):** COLLISION, robot–obstacle or robot–robot, scored on the swept path with
  stop at the first collision. It is tested paired, filter on (locked α) against filter off on
  identical layouts, by an exact McNemar test per candidate density. The effect is the paired risk
  difference with a Newcombe (method 10) 95 % CI.
* **Key secondary:**
  * INFEASIBLE (filter on).
  * The composite COLLISION-or-INFEASIBLE.
  * GOAL.
  * TIMEOUT and the stalled flag.
  * Formation error: per-trial mean slot error, analysed with Wilcoxon signed-rank and
    Hodges–Lehmann.
* **Descriptive:**
  * Interventions per trial and as a fraction of robot-steps.
  * Minimum true obstacle and robot–robot barrier values.
  * Episode length.
  * Robot–obstacle and robot–robot collisions separately.
  * Near-goal timeouts (final centroid distance ≤ 0.30 m), as defined in Phase 1.
* **Brake comparator:** the same endpoints, paired with filter off and with filter on, on the same
  layouts.
* Multiplicity: the primary test is run at three densities (Bonferroni 0.05/3). Everything else is
  reported with CIs and without confirmatory claims.

## 5. Gain grid and selection rule (filter-on validation sweep; proposed)

* **Grid:** α ∈ {0.5, 1, 2, 5, 10} s⁻¹ (`filter.alpha_sweep`). Every value satisfies αΔ ≤ 1;
  α = 10 is the boundary.
* **Sweep:** the nine policies × K ∈ {1, 2, 3} × five gains, filter on, on the existing validation
  layouts (200 per density, seeds 2,000,000 + K). The filter-off runs are repeated in the same
  sweep for pairing; they are deterministic, and Phase 1 reproduced them exactly. Optional:
  K = 5, 8 as stress cells, reported but not used for selection.
* **Selection rule (to be locked before the sweep):**
  * α\* is the gain with the highest pooled filter-on validation GOAL count, pooled over the nine
    policies and K = 1, 2, 3 (5,400 trials per gain).
  * GOAL already counts COLLISION, INFEASIBLE and TIMEOUT as failures, so the rule penalises both
    unsafe and over-conservative gains.
  * Ties go to the lower INFEASIBLE count, then to the larger α.
  * Results for every gain are reported; none is hidden.
* **Lock:** α\* is committed and tagged before any final-test layout is generated. The final test
  runs filter on at α\* only.

## 6. Brake comparator (proposed definition; not implemented, not run)

* **Brake rule (B-match):** a robot is braked when its clipped nominal command violates at least
  one of its rows of the filter's constraint set at the locked α\*. That means the same barrier
  rows, margins δ, sensing and communication ranges as the QP, with no QP solved. A braked robot
  applies (v, ω) = (0, ω_nom clipped): it stops its forward motion and keeps its commanded turn.
  Otherwise the clipped nominal command is applied.
* **Rationale:**
  * It uses the same information and the same trigger as the filter, but replaces the minimal QP
    correction with a stop. The comparison therefore isolates what the QP's minimal modification
    adds over stopping.
  * With v = 0 the body disc does not move, so a braked robot cannot hit a static obstacle in that
    period. Robot–robot collisions remain possible when a neighbour is not braked.
* **Alternative for the advisors (B-stop):** (v, ω) = (0, 0).
* **Implementation:** to be written as a separate module with tests. `cbf_qp.py` and the scoring
  stay unchanged. It runs through the same protocol loop, as a third condition beside clip and
  filter.

## 7. Final test (after α\* is locked)

* **Reserved seeds** (`docs/final_test_seed_reservation.json`, range 4,000,000–4,999,999):
  * Seed = 4,000,000 + 1,000·p + K, where p is the policy index 0–8 and K the density.
  * This gives a disjoint layout set for each policy and density. The shared-set alternative
    (4,900,000 + K) is noted in the file.
  * The range is disjoint from the training (1.0–2.0 M), validation (2.0 M + K) and pilot
    (3.0 M + K) ranges and from every literal seed in the code.
  * `tests/test_seed_reservation.py` checks this and checks that no saved result records a
    reserved seed.
  * The layouts are generated once, when the final test runs.
* **Conditions:** filter off, filter on at α\*, and brake at α\*, on identical layouts.
* **Primary analysis:**
  * One McNemar test per density on the pooled pairs of the nine policies. The layouts are
    disjoint across policies, so the pairs are independent. Under H0 each policy has p10 = p01,
    and the pooled test is valid.
  * Per-policy results are reported descriptively.
* **Sample size (open decision D6):** use `results/phase1/planning/PLANNING.md`. These are
  scenarios with explicit assumptions in both directions, not estimates. Examples for
  COLLISION, power 0.8, 0.05/3, exact test:

  | p10 | p01 | pairs per density | layouts per policy |
  |---|---|---|---|
  | 0.05 | 0.01 | 402 | 45 |
  | 0.02 | 0.005 | 1,196 | 133 |
  | 0.02 | 0.01 | 3,285 | 365 |

  Compute is not the constraint. With 1,000 layouts per policy and density, the filter-off,
  filter-on and brake runs at K = 1, 2, 3 cost about 0.9 core-hours (upper bound; section 8).
* **No re-runs with changed settings.** A crash is re-run with identical settings and seeds, and
  the re-run is logged.

## 8. Compute (estimates; `results/phase1/planning/`)

Measured inputs:

* Filter-off cost per period at batch 200: 0.011–0.034 s for K = 1–8.
* One `CBFQPFilter` call at batch 200, from a timing microbenchmark on synthetic random states
  (no trial): 0.005–0.027 s.

Upper bound: every run lasts 400 periods. The brake is bounded above by the filter cost. Repeated
benchmarks varied by about ±20 %.

| plan | core-hours | wall on 4 cores |
|---|---|---|
| validation sweep, 9 policies × K 1–3 × 5 gains, filter on + brake at each gain + filter off | about 0.7 | about 11 min |
| optional stress cells K = 5, 8, same design | about 1.0 | about 15 min |
| final test, 9 policies × K 1–3 × 1,000 layouts each, off + on(α\*) + brake(α\*) | about 0.9 | about 13 min |

## 9. Lock sequence

1. Advisors approve the limits and tolerances (`approval = "approved"` in `config/study.toml`),
   the densities (D1), the endpoints (D2), the gate (D3), the gain grid and selection rule (D4),
   the brake definition (D5) and the sample size (D6). If an environment value changes, the
   checkpoints are invalid (env fingerprint) and this draft is void.
2. This protocol is locked: dated commit and tag.
3. The brake comparator is implemented with tests. Nothing is run.
4. Filter-on validation sweep: the selection rule is applied, then α\* is committed and tagged.
5. The final-test layouts are generated from the reserved seeds and evaluated once: off, on(α\*),
   brake(α\*).
6. Analysis follows section 4. Deviations are logged with their date and reason.

## 10. Open decisions (advisor agreement not yet obtained)

* D1: densities. Keep (2, 5, 8) or move to (1, 2, 3), with 5 and 8 as stress.
* D2: primary endpoint. COLLISION alone, or the composite COLLISION-or-INFEASIBLE.
* D3: the gate. Whether a revised filter-off gate applies to the policy family, and its
  thresholds (post hoc now).
* D4: the gain grid and the selection rule in section 5.
* D5: the brake comparator definition, B-match or B-stop.
* D6: the sample-size scenario, which sets the layouts per policy.
* D7: all limits and tolerances in `config/study.toml`. The goal tolerance stays 0.10 m unless
  the advisors decide otherwise for a stated physical or task reason. Phase 1 found no physical
  reason to change it (`results/phase1/PHASE1.md`, item 3).
