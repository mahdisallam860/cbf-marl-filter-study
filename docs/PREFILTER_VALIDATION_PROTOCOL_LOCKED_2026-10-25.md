# Pre-filter validation protocol: LOCKED by the author, 2026-10-25 23:40:51 +03 (UTC+03:00, Asia/Riyadh)

**Locked by the project author. Not advisor-approved.** I, the project author, chose the
margins, levels, densities, gain rule, comparator and sample allocation in this document without
advisor review, and I locked them myself. Advisor agreement has **not** been obtained. Author
locking is recorded separately from advisor approval:

* `config/evaluation_protocol_locked.toml` holds `author_lock` and
  `advisor_approval = "not obtained"`.
* Every value in `config/study.toml` keeps `approval = "pending"`.

Machine-readable settings: `config/evaluation_protocol_locked.toml`. Calculations:
`results/prefilter_lock/` (no trials). Historical draft, kept unchanged:
`docs/PREFILTER_VALIDATION_PROTOCOL_DRAFT_2026-10-25.md`. Where the two differ, this document
governs.

The safety statements stay within `docs/CBF_DESIGN.md`, Lemma 1 under A1–A9. No result under
this protocol is a claim of continuous-time safety.

## 0. Disclosures

* **Filter-off results were seen first.** The filter-off validation results had already been
  inspected before these decisions: `results/gate/`, the curriculum held-out evaluations,
  `results/gate_v2/` and `results/phase1/` (the nine actors at K = 1, 2, 3, 4, 5, 8, the timeout
  diagnosis and the exploratory tolerance sensitivity).
* **The checkpoint rule was committed before Phase 1 but after earlier results.** It went in at
  `d598055`, before the Phase 1 evaluations, but after the gate_v2 and curriculum held-out
  validation results had been seen.
* **Validation layouts are development data.** The existing validation layouts (seeds
  2,000,000 + K) are deliberately reused for gain tuning. They support no confirmatory claim.
* **Only the reserved block supports the final evaluation.** That is the untouched, never-generated
  block 4,000,000–4,999,999 (section D6).
* **Scope of the inference.** The claim concerns the **average effect over these nine fixed
  actors**, with equal weights. It does not extend to arbitrary MAPPO policies, to independent
  training runs in general, or to every individual actor. Per-seed and per-policy results are
  reported, but a pooled usefulness result does not establish usefulness for every policy.
* **What was not run.** No filter-on, B-match, gain-selection or final-test trial has been run as
  of this lock.

## 1. Unchanged elements (D7)

* Goal tolerance 0.10 m (centroid to goal) and formation tolerance 0.10 m (mean slot error).
* Velocity box v ∈ [−0.5, 0.5] m/s, |ω| ≤ 1 rad/s. All radii and ranges.
* Control period Δ = 0.1 s, horizon 400 periods.
* Reward and training configuration, including `obstacles.densities = [2, 5, 8]`, which is the
  training mixture.
* The frozen actors and the env fingerprint `c60352aee3f11ed8`. Evaluation densities are recorded
  separately in `config/evaluation_protocol_locked.toml`; `config/study.toml` is unchanged.
* The CBF rows, margins and QP objective, collision scoring, the GOAL definition, outcome
  precedence and termination.

The non-inferiority margins (5 percentage points of GOAL, 0.01 m of formation error) are analysis
decisions. They do not change the simulator's GOAL definition or any physical parameter.

## D1. Densities and fixed actors

| role | K | data | claims |
|---|---|---|---|
| primary | 1, 2, 3 | development: validation layouts; final: reserved block | confirmatory usefulness (D2), one claim per density |
| diagnostic | 4 | validation layouts | none; does not affect gain selection |
| stress | 5, 8 | validation layouts | none; does not affect gain selection |

The nine actors are fixed in `results/phase1/checkpoint_selection.json`:

* Seed 0: iterations 526, 626, 732.
* Seeds 1 and 2: iterations 506, 606, 712.

Their file SHA-256 hashes, actor digests and provenance notes are kept (seed 1's iterations 606
and 712 come from its resumed run). No actor is replaced, selected or dropped on performance.

## D2. Confirmatory usefulness, per primary density

Each primary density is tested separately (K ∈ {1, 2, 3}). The filter at α\* is **useful** at K
only if (a), (b) and (c) all pass. The construction is intersection-union within the density, with
each component tested one-sided at α_K = 0.05/3. That level accounts for the three density
claims. There is no further division for the conjunction of three components.

Notation:

* n = 9,000 pairs per density: nine policies × 1,000 layouts. A pair is one layout run with the
  filter off and with the filter on.
* Equal numbers per policy make pooling equivalent to equal policy weights. The analysis checks
  that the counts are equal and refuses to run otherwise. A crashed run is re-run with identical
  settings.

### (a) Reduction in COLLISION-or-INFEASIBLE

* **Event:** F = outcome ∈ {COLLISION, INFEASIBLE}.
* **Counts:** n10 = #(F_off and not F_on), n01 = #(F_on and not F_off).
* **Test:** one-sided exact McNemar on the pooled pairs. H1: filter on fails less often.
  p = P(X ≥ n10) for X ~ Binomial(n10 + n01, 1/2); p = 1 when n10 + n01 = 0.
* **Pass:** p ≤ 0.05/3.
* **Estimate:** Δ_F = P(F_on) − P(F_off) = (n01 − n10)/n. Negative means fewer failures with the
  filter.
* **Interval:** Newcombe (1998) method 10 for paired proportions, which combines Wilson score
  intervals with the correlation φ, without continuity correction.
  * The one-sided 98.333 % upper bound is the upper limit of the two-sided 96.667 % interval.
  * The ordinary two-sided 95 % interval is reported descriptively.
  * Coverage is approximate.
* **Stratification:** the pooled pairs are treated as one sample. With equal policy allocation
  this ignores the policy strata. It can only overstate the variance of the equal-weight mean
  (between-policy spread adds to the pooled variance), so the interval is conservative with
  respect to stratification.

### (b) Non-inferiority of GOAL completion (absolute margin 0.05)

* **Loss:** Λ_G = P(GOAL, off) − P(GOAL, on) = (g10 − g01)/n, where
  g10 = #(GOAL_off and not GOAL_on) and g01 = #(GOAL_on and not GOAL_off). Positive means the
  filter loses GOAL completions.
* **Every pair is kept.** An INFEASIBLE arm has GOAL = 0. A pair is discordant only if exactly one
  arm reaches GOAL; no pair is assigned a loss of 1 automatically.
* **Bound:** Newcombe method 10 for P(GOAL, on) − P(GOAL, off). The one-sided 98.333 % upper
  bound of Λ_G is −(lower limit of the two-sided 96.667 % interval).
* **Pass:** the upper bound ≤ 0.05. The margin is five percentage points absolute, not a 5 %
  relative loss.
* A two-sided 95 % interval is reported descriptively. Coverage is approximate, and
  stratification is handled as in (a).

### (c) Non-inferiority of common-prefix formation error (margin Y = 0.01 m)

Executed periods:

* T_arm is the number of control periods actually executed by that arm. It equals the protocol's
  `end_step`:
  * For GOAL and COLLISION, the trial ends after the period in which the event occurred, so that
    period counts.
  * For INFEASIBLE, the trial stops at the attempted period before any motion, so that period
    does not count.
  * For TIMEOUT, T_arm = 400.
* The protocol's `active` flag is not used, because it is set before the infeasibility check.

Per-period error:

* e_arm(t) is the existing per-step mean formation slot error (`protocol.formation_error`) at the
  post-step state of period t, for t = 1..T_arm.
* The initial state (t = 0) is excluded, and so are the frozen post-termination samples
  (t > T_arm).
* The collision-ending period is included.
* An attempted period that infeasibility prevented is excluded, because it is never executed.
* These are recorded from the environment's realised poses through a pass-through recorder. The
  per-trial `form_err_mean` field of the protocol is not used: it omits the collision-ending
  period.

Pair difference:

* L_i = min(T_off, T_on).
* For L_i > 0: d_i = (1/L_i) Σ_{t=1..L_i} [e_on(t) − e_off(t)].
* A pair with L_i = 0 is **excluded** with a recorded reason ("off arm executed 0 periods",
  "on arm executed 0 periods", or both). It is never counted as zero error. Such pairs remain in
  (a) and (b).

Estimate:

* For each policy p, D̄_p is the mean of d_i over its eligible pairs.
* The estimate is D̄ = (1/9) Σ_p D̄_p: equal weights, regardless of how many pairs each policy has
  eligible.

Bound:

* **Primary: stratified paired percentile bootstrap.**
  * B = 20,000 replicates, generator `numpy.random.default_rng(20261026)`.
  * In each replicate, every policy's eligible d_i are resampled with replacement (its own count
    is kept), the policy means are recomputed and averaged with weight 1/9.
  * The one-sided 98.333 % upper bound is the empirical quantile at 1 − 0.05/3 of the 20,000
    replicate estimates, with `numpy.quantile` method "linear" (Hyndman–Fan type 7).
  * The descriptive 95 % interval uses the 0.025 and 0.975 quantiles of the same replicates.
  * Pairs are resampled as units, and the policy strata and weights are fixed. The nine actors are
    the population of inference; the three training seeds are not resampled.
  * Coverage is approximate; it relies on large-sample validity of the bootstrap for the mean.
* **Cross-check (reported, not decisive):** the stratified normal bound
  D̄ + z_{0.98333} · sqrt(Σ_p s_p² / m_p) / 9, where s_p is the sample SD and m_p the eligible
  count. If the cross-check and the bootstrap disagree on pass/fail, the bootstrap decides and the
  disagreement is reported.

Pass and inconclusive:

* **Pass:** the bootstrap upper bound ≤ 0.01 m.
* **Inconclusive:** any policy has no eligible pair, or fewer than 2. With one pair its variance
  cannot be estimated and the bootstrap would treat it as known; I set this minimum before any
  data. The policy is not dropped and the weights are not renormalised. An inconclusive (c) means
  "useful" cannot be declared.
* Reported for every policy: eligible and excluded counts, the exclusion reasons, and the
  distribution of L_i (min, quartiles, max).

Interpretation:

* This is **formation preservation over the common executed prefix**.
* It does not establish preservation over complete trajectories or over a fixed full horizon.
* When the off arm collides, the prefix ends at that collision and leaves out most of the filter's
  avoidance manoeuvre. See the full-trajectory secondary below.

### Separate analyses

The mean-based non-inferiority in (c) is separate from the Wilcoxon signed-rank and
Hodges–Lehmann shift analyses, which are secondary (`cbf_study.stats`). The HL interval uses the
signed-rank normal approximation.

Per seed and per policy:

* (a), (b) and (c) are reported with two-sided 95 % intervals: Newcombe per stratum, and for
  formation a percentile bootstrap within the stratum (B = 20,000, seed 20261026).
* They are descriptive and carry no usefulness verdict.

## Secondary endpoints (pre-declared; none independently confirmatory)

1. **GOAL** under the unchanged tolerances (0.10 m centroid, 0.10 m formation): rates by arm,
   paired.
2. **Reached-goal-region:** the centroid is within 0.30 m of the goal at some recorded state
   before any failure.
   * Admissible states: t = 0..T_arm for GOAL, TIMEOUT and INFEASIBLE; t = 0..T_arm − 1 for
     COLLISION, since the collision period's end state is not before the collision.
   * This event does not change the outcome or the termination, and a later failure is still
     recorded.
3. **Final centroid-to-goal distance:** paired Wilcoxon signed-rank and Hodges–Lehmann
   (on − off).
4. **Common-prefix formation shift:** Wilcoxon and Hodges–Lehmann of the eligible d_i, separate
   from criterion (c).
5. **Full-trajectory mean formation error,** limited to pairs where **both arms reach GOAL.**
   * Each arm's error is its mean of e(t) over t = 1..T_arm, all executed periods.
   * Reported descriptively (mean, median, difference, HL), with the number of such pairs per
     policy. It is conditional on joint success and supports no confirmatory claim.
6. **Filter-on versus B-match at α\*:** on the same pairs and the same endpoints ((a), (b) and (c)
   definitions with B-match in place of filter off, and the secondaries above), with CIs.
   Nothing confirmatory follows from it.
7. **TIMEOUT and the stalled flag:** rates by arm, paired McNemar, descriptive.
8. **Interventions and brake activations:**
   * Filter interventions per trial and as a fraction of executed robot-periods.
   * B-match activations per trial and as a fraction of executed robot-periods.
9. **Episode length:** T_arm by outcome.
10. **Collisions by kind:** robot–obstacle and robot–robot separately.
11. **Barrier minima.** Two kinds are reported, with their sampling convention:
    * `h_obst_min` / `h_rr_min` from the protocol are the minimum true barrier values over the
      **pre-step states of executed and attempted periods**. They omit the terminal state and are
      not whole-trajectory minima.
    * An additional **sampled** diagnostic gives the minimum over all recorded control-instant
      states t = 0..T_arm, terminal state included. It is computed from the recorded poses and
      leaves scoring and outcomes unchanged.
    * Both are samples at control instants. Neither is a certified continuous-path bound; only the
      swept-path collision scoring of `protocol.score_motion` covers between-sample motion.

## D3. No gate on the fixed family

There is no pass/fail gate that selects or drops actors. The filter-off rates per policy and
density are reported (Phase 1 already reports them on validation). The original pooled gate (GOAL
≥ 50 % over K = 2, 5, 8) failed for every checkpoint; that disclosure stands. No revised threshold
is introduced.

## D4. Gain grid, selection and analytical prediction

* **Grid:** α ∈ {0.5, 1, 2, 5, 10} s⁻¹.
* **Development sweep:** the nine actors on the existing validation layouts (200 per density,
  seeds 2,000,000 + K), with K = 1, 2, 3 (selection) and K = 4, 5, 8 (reported only). Conditions
  per gain: filter off, filter on at α, and B-match using the rows of the same α.
* **Selection of α\*:**
  1. The highest pooled filter-on GOAL count over the nine actors and K = 1, 2, 3.
  2. Ties: fewer INFEASIBLE outcomes.
  3. Remaining ties: the larger α.
  4. K = 4, 5 and 8 do not affect the selection.
* **Reporting:** every gain is reported on every endpoint, including reached-goal-region.
* **Lock of α\*:** α\* is committed and tagged before any final-test layout is generated. It is not
  replaced after final-test results are seen. If α\* fails the usefulness criteria, that result is
  reported.

**Analytical prediction** (prediction, not observation; `results/prefilter_lock/PRECISION.md`,
from the unchanged margin functions):

* Under the tightened robot–robot rows, the barrier-point separation at which the zero command
  becomes admissible is 0.668 m at α = 0.5 and 0.501 m at α = 1 (0.408, 0.345 and 0.323 m at
  α = 2, 5, 10).
* The nominal centre-to-centre triangle spacing is 0.520 m.
* Barrier-point separation depends on the headings: it equals centre spacing only for equal
  headings and can differ by up to 2ℓ = 0.10 m.
* α = 0.5 is therefore expected to conflict strongly with holding nominal spacing.
* α = 1 may cause interventions at compressed or heading-dependent separations, but does not
  necessarily require deformation at nominal equal-heading spacing.
* These thresholds concern the feasibility of the zero command. They are not a guaranteed minimum
  physical separation.

## D5. B-match comparator

Module `cbf_study/bmatch.py`, parameterised by the gain whose rows it uses. At each robot and
period:

1. The frozen actor's command is clipped to the unchanged velocity box.
2. The robot's enabled barrier rows are built exactly as in `CBFQPFilter` at that gain: the same
   obstacle and robot–robot rows, margins δ, equal responsibility split, sensing and communication
   ranges.
3. If the clipped command violates any enabled barrier row, the robot applies v = 0 and
   ω = the clipped nominal ω.
4. Otherwise it applies the clipped nominal command.

Further rules:

* **Violation test:** a row is violated when (b − a·u)/|a| > `solver_tol`. This is the
  normalised feasibility convention of `qp2d.solve_qp2d`, including its treatment of degenerate
  rows. The intervention tolerance is not used for this test.
* **No QP.** B-match solves no QP. It is not claimed to satisfy the CBF rows in general: rotation
  moves the offset point, and another robot can collide with a stopped robot.
* **Execution:** a command adapter plugs into the unchanged protocol loop, on the clip path. The
  braked command lies inside the box, so clipping leaves it unchanged. Collision, GOAL, outcome
  and termination semantics are unchanged.
* **Logging:** the adapter records the actor's raw command, the actor's clipped command, the
  applied B-match command and the activation flag separately.
  * The protocol's own `raw` and `clipped` fields under this condition hold the B-match command.
    They are relabelled and never reported as the actor's output.
  * Brake activations come from the adapter, never from the protocol's filter-off intervention
    count.
  * The status is "no QP", never "QP feasible".
* **Gain:** in the development sweep B-match uses the gain of the filter-on condition it is
  compared with. In the final test it uses α\*.

## D6. Sample size and reserved data

**Final test:** 1,000 layouts per actor and primary density. Each actor and density has its own
disjoint seed (4,000,000 + 1,000·p + K, `docs/final_test_seed_reservation.json`), which gives
9,000 pairs per density. The same layouts are used for filter off, filter on at α\* and B-match
at α\*. The layouts are not generated or inspected until α\* is tagged. The reserved IDs for
K = 4, 5, 8 stay reserved and unused by this protocol.

The allocation is justified by precision scenarios (`results/prefilter_lock/PRECISION.md`).
Examples at n = 9,000 and one-sided 0.05/3:

* **GOAL loss.** With discordance 0.20 and a true loss of 0.03, the expected bound is 0.040 and
  P(pass) = 0.98. With a true loss of 0.04 (close to the margin), P(pass) = 0.50.
* **Formation.** With a per-pair SD of 0.02 m, 50 % of pairs excluded and a true increase of
  0.008 m, the standard error is 0.00030 m and the expected bound is 0.0086 m. Exclusions,
  variability and proximity to the margin are explicit in the grid.
* **Composite failure.** Explicit (p10, p01) grid. Power is at least 0.96 whenever
  p10 − p01 ≥ 0.005 for p10 ≤ 0.01, and it is near zero when p10 = p01. Collision prevention is
  not inferred from filter-off collisions alone, and preventing a collision is not assumed to
  prevent infeasibility.

This is a precision plan, not a guarantee of sufficiency. An inconclusive bound stays
inconclusive: it does not authorise changing a margin or adding test data.

**Method-study block:** 5,000,000–5,999,999 is reserved for a possible later method study. It is
recorded in the reservation file and checked for disjointness by `tests/test_seed_reservation.py`.
No layout is generated. A method developed after inspecting the Phase 2 findings must use its own
untouched evaluation block from this range and disclose the earlier findings used in its
development.

## Lock sequence from here

1. This protocol is committed and tagged.
2. B-match, the common-prefix metric, the recording and the statistical helpers are implemented
   with synthetic unit tests, in a separate commit. Nothing is run.
3. The development sweep on validation layouts starts only on the author's explicit go-ahead.
   α\* is then committed and tagged.
4. The final-test layouts are generated from the reserved block and evaluated once, followed by
   the analysis in D2 and the secondaries.
5. Every deviation is logged with its date and reason.
