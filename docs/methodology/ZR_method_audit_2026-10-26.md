# ZR Method Audit: CBF–MARL Filter Study

Oct 26, 2026 · @mahdi

## Decision summary

**Decision: C. Drop the methodological novelty claim and finish the empirical paper.** Keep ZR, at most, as a short derivation remark in a discussion or appendix.

**Inspected:** branch `claude/elegant-hopper-v8jxlv` at commit `df788126a054418bba8494459b173c9fe5a13329`. The audited report describes `d34c2c3`. The audit had web access. Nothing in the 5,000,000 block was touched. Saved final-test files from the already-used 4,000,000 block were read and used for arithmetic only. The checks were standalone scripts that import nothing from the study.

**Why C:**

1. **The mathematics holds in exact arithmetic, but it is a specialization, not a new result.**
   - The one-sided lemma is Theorem 3 of [Breeden, Garg, Panagou (L-CSS 2022)](https://arxiv.org/pdf/2103.03677), with linear α = γ/T and γ ≤ 1.
   - That proof uses the supremum over U only to bound ψ at the applied input. Evaluating the bound at u\_k is therefore an immediate corollary.
   - Zero command as the feasibility witness for decentralized, equal-split constraints under held commands is in [ORCA](https://ics-websites.science.uu.nl/docs/vakken/mcrws/papers_new/van%20den%20Berg%20et%20al%20-%202011%20-%20Reciprocal%20n-body%20collision%20avoidance.pdf), Sec. 5.2 (holonomic case, v\_opt = 0).
   - Stopping as the backup that guarantees feasibility is in [Chen, Singletary, Ames (L-CSS 2021)](https://par.nsf.gov/servlets/purl/10281485).
   - What remains is a two-row bound in |ω| for the unicycle offset point that keeps the QP: a useful derivation, not a contribution.
2. **The guarantee does not survive floating point.** In the completed final test, every filter-on failure was an INFEASIBLE QP (433 trials). At the infeasible states recomputed from the saved logs, robots sit on the current filter's α = 10 stand-off. The zero-command slack ranges from −3.7e-7 to +7.5e-5, median −1.5e-8. With αΔ = 1 the filter reaches its stand-off in about one step and then pins there, so float32 noise decides feasibility when rows oppose. ZR pins at h = 0 in the same way and would need the same tolerance rule, and that rule would also fix the current filter.
3. **No completed-study verdict could change.** ZR's best-case gain is the infeasible share: 0.07, 0.83 and 3.9 points at K = 1, 2, 3. The GOAL criterion misses by 0.31, 5.6 and 6.3 points, and the formation criterion fails at every density regardless. The most likely effect is converting INFEASIBLE trials into stalled TIMEOUTs; 61% of filter-on timeouts are already obstacle-front stalls.
4. **Against the justified fallback, the edge is cosmetic.** The established one-sided margin (B) with a full stop (0, 0) fallback is also certified. Its commands always lie inside ZR's admissible set. ZR's remaining advantage is no switching and a command at least as close to the nominal at the same state; it is not expected to be measurable.

**Defensible statement, if any (remark only):** for the offset-point unicycle barriers of this study, Breeden et al.'s second-order sampled-data margin, evaluated at the held command, can be bounded by two affine rows per barrier. Zero command is then admissible whenever h ≥ 0, and the decentralized equal-split filter is recursively feasible and intersample-safe, in exact arithmetic only.

## 1. Mathematics, re-derived

The construction is correct in exact arithmetic; its implementation guarantee fails at float32 precision, and one test trial lies outside its premise.

**Held-input kinematics.** Under a held command:

- ṗ = v·e(θ) + ℓω·e⊥(θ), so s = |ṗ| = √(v² + ℓ²ω²) is constant.
- p̈ = ω·Rot(π/2)·ṗ, so |p̈| = |ω|s. The offset point moves on a circle at constant speed.

**One-sided lemma (verified; existing result).** Assume:

- u is held on \[t\_k, t\_k + Δ\];
- ḧ ≥ −M on that interval, with M ≥ 0;
- αΔ ≤ 1 and h(t\_k) ≥ 0;
- ḣ(t\_k) + αh(t\_k) ≥ MΔ/2.

Then h(t\_k + s) ≥ (1 − αs)h(t\_k) + (Ms/2)(Δ − s) ≥ 0 for every s in \[0, Δ\]. It needs h twice differentiable along the arc (true here), M ≥ 0, h(t\_k) ≥ 0 and αΔ ≤ 1. This is Breeden et al., Theorem 3, Eqs. (17)–(18) of Sec. III.

**Obstacle bound (verified).**

- ḧ = 2s² + 2(p − c)·p̈ ≥ 2s² − 2(d\_k + sΔ)|ω|s.
- Maximize over σ in \[0, V\_p\]: F(|ω|) = max over σ of \[2(d + σΔ)σ|ω| − 2σ²\]₊. This is a pointwise maximum of functions affine in |ω|, so it is convex, and F(0) = 0. The chord bound F(|ω|) ≤ (|ω|/ω̄)F(ω̄) follows.
- The complete coefficient, which the report gave only for the first branch:

| Distance range | κ\_o(d) |
| --- | --- |
| D\_o ≤ d ≤ d\* = 2V\_p(1 − Δω̄)/ω̄ = 0.9045 m | Δd²ω̄ / (4(1 − Δω̄)) |
| d\* < d ≤ sensing range (1.0 m) | (Δ/ω̄)\[(d + V\_pΔ)ω̄V\_p − V\_p²\] |

- The two branches agree at d\* (0.02272). The coefficient runs from 0.0034 at d = 0.35 to 0.0275 at d = 1.0.
- Units: κ is in m², so κ|ω| is in m²/s, matching ḣ. The construction needs Δω̄ < 1 (here 0.1).

**Pair bound (verified, including when only one robot turns).**

- ḧ\_ij ≥ −2(d + (s\_i + s\_j)Δ)(|ω\_i|s\_i + |ω\_j|s\_j) ≥ −2(d + 2V\_pΔ)V\_p(|ω\_i| + |ω\_j|).
- κ\_r(d) = ΔV\_p(d + 2V\_pΔ), from 0.0201 at d = 0.30 to 0.0804 at d = 1.5 m.
- If ω\_j = 0, robot j's row carries no margin, but j's speed still enters through the distance growth, which the 2V\_pΔ term covers.

**Summing the split rows (verified).** Adding robot i's and robot j's rows gives ḣ\_ij + αh\_ij ≥ κ\_r(|ω\_i| + |ω\_j|) ≥ (Δ/2)M\_ij. This requires:

- both robots use the same state at t\_k (A3);
- both rows are enabled; the mask |p\_i − p\_j| ≤ comm range is symmetric.

**Absolute value as two affine rows (verified).** a·u − κ|ω| ≥ b holds exactly when a·u − κω ≥ b and a·u + κω ≥ b, provided κ ≥ 0.

**Feasibility, recursion and range masks (verified in exact arithmetic).**

- u = 0 satisfies every enabled row when h ≥ 0, and the box contains 0.
- Barriers omitted by the range rule cannot reach zero within a period: sensing range 1.0 ≥ 0.40 and comm range 1.5 ≥ 0.40.
- Induction gives recursive feasibility and intersample safety under A1–A8.

**Synthetic checks (own exact arcs, no repository code):**

| Check | Result | Required |
| --- | --- | --- |
| Obstacle bound, 200,000 random states including d < D\_o | worst ratio 0.90 | ≤ 1 |
| Pair bound, 30% with ω\_j = 0 | worst ratio 0.89 | ≤ 1 |
| Theorem, obstacle: 2.4 million admissible commands, h(0) in \[0, 0.01\], α ∈ {0.5, 2, 10}, random headings including tangent | min of h(t) − h₀(1 − αt) = −8e-17 | ≥ 0 up to rounding |
| Theorem, pair: 3.6 million command pairs, including head-on and mixed turning | −6e-17 | ≥ 0 up to rounding |

**Constructed cases:**

- **Tangent heading at h = 0.** a = (0, −0.035). v is unrestricted; ω is allowed only in the outward direction.
- **Stand-off (is u = 0 admissible?).**

| Filter | h = 0 | h = 1e-4 | h = 0.004, α = 10 |
| --- | --- | --- | --- |
| Current filter (C) | no | no | no |
| One-sided state-only (B) | no | no | yes |
| ZR | yes | yes | yes |

**Exact arithmetic versus implementation:**

- The QP solver's tolerance is 1e-9 on normalized rows. A float32 tracking error of 6e-8 m makes h about −4.2e-8. The zero-command row is then violated by about 1.2e-5 at α = 10, so feasibility is not guaranteed in float32.
- A robust version needs either a stated tolerance rule, or a tracking-error model that vanishes at u = 0 plus an extra term proportional to the command. Whether the simulator holds exactly still at u = 0 is not verified.

**Trial 646 (s2\_early, K = 3) is outside the premise.**

- Robot 2 starts with h(0) = −0.00777: its barrier point is 0.3387 m from the obstacle centre against D\_o = 0.35. The body is clear (axle surface gap 0.087 m).
- The layout sampler checks robot–robot spacing, but checks obstacle clearance only against the nominal start geometry (0.75 m from the start centroid). This obstacle sits at 0.7695 m, and the position noise is about 2.6σ.
- With h(0) < 0, the zero-command row requires −αh(0) ≤ 0, which fails. The lemma then bounds h(t) only by the negative value h(0)(1 − αt). Neither the original Lemma 1 nor ZR's theorem certifies that start.
- It is the only such trial in 27,000. Keep it unchanged and disclose it as an exception to assumption A5.

## 2. Originality against primary sources

Every ingredient of ZR appears in prior work; only the two-row |ω| bound for this unicycle barrier is specific to it.

| Source (location) | Overlap | Remaining difference | Category |
| --- | --- | --- | --- |
| [Breeden et al. 2022](https://arxiv.org/pdf/2103.03677), Thm 3, Eqs. 17–18 | The one-sided lemma; the proof uses the supremum over U only at the applied u\_k | Input-dependent η with η(x, 0) = 0; affine \|ω\| bound | Existing idea; ZR is a specialization |
| [Singletary et al. 2020](https://arxiv.org/pdf/2005.06418), Remarks 3–4 | Considered an input-dependent check and declined it because it is not affine; infeasible boundary handled by a backup | ZR keeps an input-dependent bound affine for this one system | Derivation |
| [van den Berg et al. 2011 (ORCA)](https://ics-websites.science.uu.nl/docs/vakken/mcrws/papers_new/van%20den%20Berg%20et%20al%20-%202011%20-%20Reciprocal%20n-body%20collision%20avoidance.pdf), Eq. 6, Sec. 5.2 | Decentralized half-responsibility; held velocities are collision-free for τ; with v\_opt = 0 the origin lies in every half-plane, so a velocity always exists | Holonomic straight-line motion, no barrier decay, no curvature bound | Existing idea |
| [Chen, Singletary, Ames 2021](https://par.nsf.gov/servlets/purl/10281485), Prop. 1, Thm 1 | Feasible on {h ≥ 0} because the backup (brake) input is admissible; decentralized 0.5 split | Continuous time (grid sampling), Dubins model with acceleration input | Existing idea |
| [Yang, Belta, Tron 2019](https://arxiv.org/pdf/1903.03692), Sec. IV | Intersample bounds that depend on the held input | Used to choose trigger times, not to shape constraints | Existing idea |
| [Alonso-Mora et al. 2010 (NH-ORCA)](https://la.disneyresearch.com/wp-content/uploads/alonsomora10dars_paper1.pdf), Secs. 2–5 | Unicycle reciprocal avoidance; zero input as fallback when infeasible | No barrier condition; feasibility not proved | Close relative |
| [Singletary, Kolathaya, Ames 2021](https://arxiv.org/pdf/2009.09100), Lemma 2 | Driftless (velocity-input) CBF; u = 0 satisfies it when h ≥ 0 (implied, not stated) | No sampling | Existing idea |

All of these use margins that do not depend on the applied input:

- [Cortez et al.](https://arxiv.org/pdf/1903.09816), Eq. 18;
- [Tan, Daş, Ames, Burdick (ZOCBF)](https://arxiv.org/html/2411.17079v2), Lemma 1;
- [Liu, Xiao, Belta (SACBF)](https://arxiv.org/html/2511.11897v2), Eq. 25;
- [Zhang, Walters, Xu (interval CBF)](https://arxiv.org/pdf/2110.00915), Eqs. 12–14;
- [Taylor, Ong, Cortés, Ames (event-triggered)](https://arxiv.org/pdf/2003.06963), Thm 1, using a uniform bound F.

**Coverage gaps (uncertainty, not evidence of originality):**

- [Taylor et al. 2022](https://arxiv.org/abs/2203.11470v2): abstract only.
- [Choi et al. 2025](https://arxiv.org/abs/2505.02293): partial text.
- [Wang, Ames, Egerstedt 2017](https://authors.library.caltech.edu/records/tshzw-g4v69): abstract only.
- [Gao et al. 2025](https://www.researchgate.net/publication/398910195_Safety-Critical_Finite-Time_Cooperative_Control_for_Sampled_Multi-Robot_Systems) and an Advanced Robotics 2026 article: inaccessible.

**Classification:**

- **Existing ideas:** the lemma; zero command as feasibility witness; input-dependent intersample bounds.
- **Useful specialization:** the affine |ω| chord bound for the offset-point unicycle, and the exact-arithmetic recursive-feasibility statement.
- **Potentially original theorem:** none that stands alone. The theorem follows in a few lines from Breeden et al.'s proof plus an ORCA-style witness.

## 3. Comparator and claim corrections

The one-sided margin with a full stop fallback is certified and always ZR-admissible; B-match is not a certified stop, and the report's strict-inclusion and "tightest" claims need correcting.

**B plus full-stop fallback.** Each robot solves the B QP:

- obstacle rows use the margin κ\_o(d)ω̄;
- each robot's pair share is κ\_r(d)ω̄;
- if the QP is infeasible, the robot applies (v, ω) = (0, 0) for that period.

Safety (verified analytically and on synthetic cases, min of h(t) − h₀(1 − αt/2) = −6e-17):

- **Robot stopped next to an obstacle.** The robot does not move, so h stays constant. This assumes the simulator holds exactly still at u = 0.
- **One robot stopped, the other on its QP.** The moving robot's own row alone satisfies the lemma with decay α/2.
- **Both stopped.** h stays constant.

The stop never needs a constraint row to hold, so it is more robust numerically than any QP-only scheme. Its commands always lie inside ZR's admissible set: B solutions are in K\_B, which is contained in K\_ZR, and 0 is in K\_ZR. So ZR's command is pointwise at least as close to the nominal.

**B-match is not a certified stop.** Under (0, ω\_nom), the body disc does not move, because it is centred on the axle, but the offset point rotates. Starting from h = 0, one period at ω = +1 leaves the barrier set with h = −0.0035 m², so the premise h ≥ 0 is lost.

**Containment.** The admissible sets satisfy K\_C ⊆ K\_B ⊆ K\_ZR, non-strictly, for the same enabled rows and the same α.

- The report's "strictly" is false. Counterexamples: no enabled rows; or one obstacle at d = 1.0 with α = 10, where every point of the box gives a·u ≥ −1.1 while every margin's bound is at most about −8.70, so all three sets equal the box.
- Strictness holds, for example, when 0 ≤ αh < m\_C for some enabled barrier (so K\_C ⊊ K\_ZR), or 0 ≤ αh < m\_B (so K\_B ⊊ K\_ZR).

**Wording.**

- B is the maximum within one specific bounding chain, not the exact worst case of −ḧ; drop "tightest".
- B plus stop is the strongest only among the same-information, single-step alternatives considered here; drop "strongest" without that scope.

## 4. Scientific value

ZR's theoretical value is small and its expected navigation value is essentially zero.

- **Theoretical value:** small. It is a clean remark that keeps stopping admissible under the existing kind of guarantee, but it is not a new method.
- **Navigation value:** essentially none expected. At the 433 recorded infeasible states, B already admits stopping with slack 0.042. The events are float-level pinning that a tolerance rule removes for every margin. ZR would mostly convert INFEASIBLE into stalled TIMEOUT. It also leaves untouched the real cost found by the mechanism analysis: obstacle-front stalls on paths the unfiltered policy completed.

**Smallest distinguishing comparison (not launched).**

- **Data:** development data only (validation layouts), K = 3, α = 10, the nine actors.
- **Arms:** C + tolerance τ, B + τ, ZR + τ, and B + full stop.
- **Measures:** INFEASIBLE, GOAL, and stall classes.
- **Reading:** if C + τ and B + τ already have near-zero INFEASIBLE and GOAL within ±1 point of ZR + τ, ZR adds nothing measurable.
- Any confirmatory follow-up must use the 5,000,000 block and disclose that these post-test analyses informed it.

## 5. Claims table

The exact-arithmetic claims hold; the implementation, strictness, wording and originality claims do not stand as written.

| Claim | Status |
| --- | --- |
| One-sided lemma | Verified; existing (Breeden Thm 3) |
| Obstacle bound, speed maximization, chord, two-branch coefficient | Verified (report's coefficient incomplete beyond d = 0.905 m) |
| Pair bound, including one robot turning; split rows sum to the pair condition | Verified, given A3 and the symmetric mask |
| \|ω\| written as two affine rows | Verified |
| Zero command feasible; recursive feasibility; intersample safety | Verified in exact arithmetic under A1–A8 |
| No INFEASIBLE in the implementation | Conditional on a tolerance or robustness rule; fails in float32 |
| Theorem covers the test set | Conditional on h(0) ≥ 0; violated in 1 of 27,000 (trial 646) |
| Universal strict inclusion | False (counterexamples in section 3); non-strict holds |
| "Tightest" / "strongest" | False as stated; needs the scope given in section 3 |
| Originality "moderate" (report) | Revised to low |
| ZR improves GOAL or stalls | Unresolved; expected about zero; at most 0.07 / 0.83 / 3.9 points |
| Robust version for VMAS tracking error | Unresolved |

**Counterexamples found:**

- set equality far from obstacles (strictness is false);
- float32 noise at h ≈ 0 breaks ZR's zero-command feasibility;
- trial 646, an h(0) < 0 start;
- B-match leaving the barrier set at h = 0.

**Independent verification still needed:**

- a second person's check of the lemma, the chord step and the pair proof;
- the unrounded slack at all 433 states (273 robots in three actors' K = 3 cells were sampled);
- whether VMAS holds exactly still at u = 0;
- the coverage gaps listed in section 2.

## 6. Bounded next step (≤ 7 days)

Document the pinning mechanism and the A5 exception, then draft the empirical paper's discussion for advisor review.

1. **Days 1–2:** Add a POST HOC script that recomputes the zero-command slack at all 433 infeasible states from the saved logs. Report the pinning mechanism (deadbeat αΔ = 1 plus float32) and the trial-646 A5 exception without changing any confirmatory number.
2. **Days 3–5:** Draft the empirical paper's discussion:
   - the obstacle-front stall mechanism;
   - the 13× margin conservatism the one-sided bound would remove;
   - the ZR derivation as a one-paragraph remark, citing Breeden Thm 3, ORCA and Chen–Singletary–Ames.
3. **Days 6–7:** Send the draft to the advisors with an explicit disclosure of which post-test findings would inform any 5,000,000-block follow-up, such as the exact-footprint variant ZR-0, which stays deferred.
