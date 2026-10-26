# CBF–MARL methodology study: research report

**Date of investigation and literature search:** 2026-10-25
**Repository:** https://github.com/mahdisallam860/cbf-marl-filter-study, branch `claude/elegant-hopper-v8jxlv`, commit `d34c2c3c5b9045618bec5bc95e30dec79839a094`
**Scope:** research only. No training, no filter-on or brake trials, no simulations, no final-test layouts, no repository changes.

---

## Summary

**Recommendation.** Pursue one small, gated method: a sampled-data CBF tightening that is proportional to the held command, so the margin vanishes at rest. Fall back to the empirical study if it does not beat the strongest prior-art margin.

- **What changes.** Each barrier row of the existing filter becomes two linear rows. The 2-variable QP and its exact solver stay as they are.
- **What it proves.** With stopping always admissible, the decentralized filter is recursively feasible, with the same intersample guarantee. Assumption A9 (every QP is feasible) becomes a theorem, so INFEASIBLE cannot occur in exact arithmetic.
- **Originality confidence.** Moderate for the specific construction, low for the underlying idea. In the sampled-data CBF papers I opened, every margin is a worst case over the whole input set U. One paper explicitly declines an input-dependent margin to keep the constraint affine ([Singletary et al. 2020](https://arxiv.org/pdf/2005.06418)). I cannot exclude a precedent in literature I could not open.
- **Strongest case for publication.** The closest works assume step-wise feasibility, fall back to a backup controller, or add slack. This construction makes feasibility a theorem for decentralized kinematic teams, with no extra information and an unchanged QP. It also makes an exact-footprint barrier usable under sampling.
- **Main risk.** The strongest fair comparator is a tight state-only one-sided margin plus a stop fallback. The gain-selection rule will probably push α to 5–10. There the margin differences shrink and the benefit may disappear.

---

## 1. Verified repository state and the actual gap

**Snapshot.** The branch `claude/elegant-hopper-v8jxlv` head is `d34c2c3c5b9045618bec5bc95e30dec79839a094`, with no later commits. `main` is at `728aaf2`. I read the nine Phase 1 files, `CBF_DESIGN.md`, `MAPPO_POLICY.md` and the listed modules in full. I skimmed `solvability.py`, `stats.py` and the scripts. Nothing was run except arithmetic on derived formulas and read-only summaries of saved CSVs.

**Implementation details verified in code:**

- **Margins.** The offset-point barriers and margins match the design doc:
  - δ_o = (Δ/2)[2V_p² + 2(d+V_pΔ)ω̄V_p];
  - each robot's robot–robot share is (Δ/4)[8V_p² + 4(d+2V_pΔ)ω̄V_p];
  - V_p = 0.5025.
- **Margin type.** Both margins are two-sided bounds on |ḧ| and independent of the command.
- **Solver.** It projects the raw nominal command onto the polygon. On infeasibility the trial stops before motion, with no fallback.
- **Stand-offs.** My closed-form values reproduce the doc's table exactly: 12.5, 3.2, 1.3 and 0.65 cm for obstacles; 36.8, 10.7, 4.5 and 2.3 cm for robot pairs.

**Discrepancies to resolve before locking:**

- The committed draft uses COLLISION as primary and Wilcoxon/Hodges–Lehmann for formation error. The current proposal uses the composite endpoint and a mean-increase non-inferiority test.
- The README still says "STOP at the filter-off gate".

**New analytical findings (derived; no simulation):**

1. **Most of the current margin's conservatism is removable with known theory.** Lemma 1 only needs a lower bound on ḧ. For obstacles, ḧ ≥ 2s² − 2(d+sΔ)|ω|s, where s = |ṗ| is constant under a held command. Maximizing over s gives the tight one-sided state-only margin. This is an analytic specialization of the second-order margin in [Breeden et al.](https://arxiv.org/abs/2103.03677).

   | | Current (C) | Tight one-sided (B) | Ratio |
   |---|---|---|---|
   | Obstacle margin, d = 0.35 | 0.0454 | 0.0034 | 13× |
   | Pair share, d = 0.52 | 0.0817 | 0.0312 | 2.6× |
   | Obstacle stand-off, α = 2 | 3.2 cm | 0.25 cm | |
   | Pair stand-off, α = 2 | 10.7 cm | 3.4 cm | |

   A reviewer can attribute GOAL loss to this avoidable conservatism. This should become an advisor decision.

2. **Stopping is inadmissible near the boundary.** For both C and B, u = 0 violates a row whenever h < m/α. A9 is therefore a real assumption. The saved stress sweep already shows the effect: 231 of 600 trials were INFEASIBLE at α = 0.5.

3. **The formation at rest is incompatible with C at small α.** The 0.30 m triangle has barrier-point spacing of about 0.52 m (h = 0.18). C admits that configuration at rest only if α ≥ 0.91. With heading differences that shrink the spacing to 0.42 m, the threshold rises to α ≥ 1.78. At α = 0.5, C forces the formation to expand or makes the QP infeasible.

4. **Equal splitting throttles formation transit, for every margin.** The slots are world-fixed, and the 1–2 pair axis lies along the travel direction. The trailing robot's row bounds its along-axis speed by [(α/2)h − m]/(2d). With ω = 0 and zero margin this is 0.087α m/s:

   | α | C | B | ZR |
   |---|---|---|---|
   | 2 | 0.10 m/s | 0.14 m/s | 0.17 m/s |
   | 5 | 0.35 m/s | 0.40 m/s | 0.43 m/s |
   | 10 | 0.79 m/s | 0.84 m/s | 0.87 m/s |

   Filter-off GOAL trials end at period 74 (median; q10 66, q90 106). That means transit at roughly 0.27–0.44 m/s. Prediction: any filter at α ≤ 2 intervenes even in open space and distorts the formation, and GOAL-maximizing selection lands at α = 5–10.

5. **Saved Phase 1 evidence on barrier crossings.** A share of filter-off GOAL trials pass through states outside the barrier sets without any body contact:

   | K | Cross robot–robot barrier | Cross obstacle barrier |
   |---|---|---|
   | 2 | 16% | 16% |
   | 3 | 26% | 25% |

   These are lower bounds, because logged minima are pre-motion samples. The median minimum robot-pair barrier-point distance is 0.34–0.44 m. Robot–robot rows will be active often. The cover-disc barrier, not only the margin, will cost task completion.

6. **Two estimand issues in `run_trials`:**
   - **Zero-length trials.** `form_err_mean` averages over periods that ended without collision, using `form_sum / max(form_cnt, 1)`. A trial that ends at period 0 (INFEASIBLE) or period 1 (COLLISION) records 0.0, the best possible value. No Phase 1 filter-off trial is affected (0 of 10,800). Filter-on INFEASIBLE at period 0, which is likely at small α, would flatter the filter on formation.
   - **Duration.** The per-trial mean is duration-dependent. Converting a mid-field collision (median period 20) into a near-goal TIMEOUT adds about 250 stationary low-error periods. That lowers the mean error mechanically.
   - **Barrier logs.** `h_*_min` uses pre-motion samples only, so it omits terminal states and intersample minima.

**Research gap.** For decentralized, sampled-data CBF filtering of frozen learned multi-robot policies, published margins are state-only. That makes stopping inadmissible near the boundary, so feasibility is either assumed or patched with fallbacks or slack. No paper I screened quantifies how margin type and responsibility splitting interact with a learned formation's resting and transit geometry. The near-goal parking stalls are a nominal-policy problem that no minimal-modification filter can fix.

---

## 2. Focused literature comparison

Search date: 2026-10-25. Depth key: **F** = full text processed by automated extraction (equations as reported, not re-derived by me); **P** = partial text; **A** = abstract or metadata only; **S** = search listing only.

| Work | Setting and mechanism | Margin and feasibility | Overlap with this project | Depth |
|---|---|---|---|---|
| [Vinod et al., ICRA 2022 (MERL)](https://www.merl.com/publications/docs/TR2022-053.pdf) | Linear and double-integrator agents. RL trained without the filter, then a centralized QP filter under ZOH. | Feasibility at every step is assumed; recursive feasibility is left to future work. 500 trials: 80.6% completed, the rest deadlocked, no infeasibility. | Closest "train without, filter at execution" design. Centralized, not decentralized; no unicycle. | F |
| [Dawood et al., arXiv 2312.12861 v3](https://arxiv.org/html/2312.12861v3) | Unicycles, centroid-goal formation, distributed nonlinear MPC shields in training and execution, MASAC. | No feasibility or deadlock guarantees; the MPC often stops robots, causing timeouts. | Same task family. It has no deployment-only baseline for a filter-unaware policy. | F |
| [Choi et al., arXiv 2505.02293 (RSS 2025)](https://arxiv.org/abs/2505.02293) | Value-function (CBVF) filter, priority to the most urgent pair, filter used in training via curriculum. | Equations were cut off in my extraction. | Conflicting constraints; different dynamics and training regime. | P |
| [Singletary, Chen, Ames, CDC 2020](https://arxiv.org/pdf/2005.06418) | Sampled-data CBF via the reachable set under all admissible inputs, plus input delays. | Remark 3 declines an input-dependent margin to stay affine. Remark 4: an infeasible region near the boundary is handled by the backup controller. | The direct contrast for the proposed method. | F |
| [Breeden, Garg, Panagou, L-CSS 2022](https://arxiv.org/abs/2103.03677) | Sampled-data margins ν₀–ν₃; ν₃ is a one-sided second-order margin. | Every margin is a supremum over U. Large margins can cause infeasibility; the unicycle example sometimes failed to reach the goal. | ν₃ specialized here is CBF-B, the closest published margin. | F |
| [Tan, Daş, Ames, Burdick (ZOCBF)](https://arxiv.org/html/2411.17079v2) | Zero-order (difference-type) CBFs. | Constant buffer δ ≥ h̄ₓMT; feasibility is built into the definition. | Margin is input-independent. | F |
| [Liu, Xiao, Belta (SACBF), arXiv 2511.11897 v2](https://arxiv.org/html/2511.11897v2) | Sampling-aware high-order CBF for a single unicycle. | The second-derivative bound takes a supremum over U. A slack term is added for feasibility. | Same gap; single agent. | F |
| [Yang, Belta, Tron, ACC 2019](https://arxiv.org/pdf/1903.03692) | Self-triggered CBF. | Inter-sample bound depends on the held input, but is used only to choose the trigger time. No feasibility guarantee. | Nearest use of input-dependent inter-sample analysis. | F |
| [Chen, Singletary, Ames, L-CSS 2021](https://par.nsf.gov/servlets/purl/10281485) | Backup-trajectory CBFs, decentralized 0.5 split, braking backup. | Feasibility follows because the backup input is always admissible. Sampling is not analyzed. | The stop-fallback comparator draws on this. | F |
| [Emam et al., CDC 2019 (Robotarium)](https://arxiv.org/pdf/1909.02966) | Offset-point unicycles with a robust CBF. | Constant input-offset disturbance; centralized QP; no feasibility analysis. | Same barrier point. | F |
| [Bahati, Ong, Ames, arXiv 2304.08685](https://arxiv.org/pdf/2304.08685) | Sample-and-hold safety via a modified explicit controller. | No optimization; requires an upper bound on the sampling time. | Different route. | F |
| [Oruganti et al., arXiv 2309.08050](https://ar5iv.labs.arxiv.org/html/2309.08050) | Robust sampled-data CBF with a unicycle example. | Margin taken over U. | Same gap. | P |
| [Singh et al., arXiv 2610.17916](https://arxiv.org/html/2610.17916) | Obstacle-centred unicycle barrier (no ω term), continuous time. | Pointwise feasibility test. Sampled-data baseline uses a worst-case margin; recursive feasibility left to future work. | Supports the exact-footprint (axle-centre) variant. | F |
| [Kim, arXiv 2603.06954](https://arxiv.org/html/2603.06954v1) | Critique of hidden assumptions in CBF safety claims. | Calls for verifying feasibility under the true input bounds. | Motivates turning A9 into a theorem. | F |
| [Wang, Ames, Egerstedt, T-RO 2017](https://authors.library.caltech.edu/records/tshzw-g4v69) | Centralized, then decentralized barrier certificates. | Relaxed CBFs; braking and perturbation for deadlocks. | Source of the equal split. | A |
| [Zeng et al., 2021 (optimal-decay CBF)](https://arxiv.org/abs/2103.12375) | Decay rate optimized pointwise. | Pointwise feasibility claimed inside the safe set. | Adaptive-gain route (rejected below). | A |
| [Zhou et al., RA-L 2017 (BVC)](https://msl.stanford.edu/bibliography/zhou_fast_2017.html) | Buffered Voronoi cells, distributed, no communication. | Proven for single integrators. | Alternative decentralized design. | A |
| [Taylor et al., arXiv 2203.11470](https://arxiv.org/abs/2203.11470v2) | Sampled-data CBFs via approximate discrete-time models. | Margin form not seen. | Uncertainty. | A |

Screened at the search-listing level only: undesired equilibria ([Reis et al.](https://arxiv.org/pdf/2003.07819); [Tan & Dimarogonas, Automatica 2024](https://people.kth.se/~dimos/pdfs/Xiao_Automatica_2024.pdf); [Grover et al.](https://arxiv.org/pdf/1911.09146)); responsibility allocation ([Lyu et al.](https://arxiv.org/pdf/2206.09030); [Cosner et al.](https://ar5iv.labs.arxiv.org/html/2303.03504); [arXiv 2410.07409](https://arxiv.org/pdf/2410.07409)); [CBF-RL](https://www.alphaxiv.org/abs/2510.14959); [Muraccioli et al. 2026](https://arxiv.org/abs/2607.14488). Not accessible: [Gao et al. 2025, IJICS](https://www.researchgate.net/publication/398910195_Safety-Critical_Finite-Time_Cooperative_Control_for_Sampled_Multi-Robot_Systems) (abstract unavailable). These gaps are uncertainty, not evidence of originality.

---

## 3. Candidate ranking

H/M/L = high/medium/low. "Risk" is the risk that the benefit disappears under fair comparison.

| # | Candidate | Mechanism addressed | Originality | Importance | Tractability | Cost | Risk | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | **ZR**: zero-at-rest, \|ω\|-proportional tightening (offset barrier) | Infeasibility, at-rest incompatibility, pair stand-offs | M (construction) / L (idea) | M | H | L | H | **Primary, gated** |
| 2 | **ZR-0**: same tightening on the exact axle-centre footprint | Cover-disc conservatism (findings 1 and 5), transit cap | M− (depends on 1) | M–H | H | L–M | M–H | Secondary arm of 1 |
| 3 | **CBF-B**: tight one-sided state-only margin | Margin conservatism (finding 1) | None: specialization of Breeden ν₃ | H for the study | H | L | n/a | Comparator and recommended upgrade, not a contribution |
| 4 | Teammate-aware robot–robot constraints (team QP over 6 variables, or consistent velocity-aware allocation) | Transit throttling (finding 4) | L (Wang 2017; responsibility-allocation work) | M–H | H | M (team QP needs nominal-command sharing) | M | Diagnostic or future work |
| 5 | Liveness and deadlock escape | Filter-induced stalls at obstacles | L (large literature) | Unknown: Phase 1 stalls are nominal and away from obstacles | M | M | H | Only if Phase 2 shows obstacle-adjacent stalls |
| 6 | Adaptive or optimal-decay α | Infeasibility | L | M | H | L | n/a | Rejected (see note) |
| 7 | Predictive or backup multi-step filter | Feasibility, foresight | L (Vinod; backup CBFs; gatekeeper) | M | M | M–H | M | Rejected |
| 8 | Terminal parking-controller handoff | Nominal near-goal stalls | Not a safety-filter contribution | H for GOAL | H | L | Confounds GOAL | Rejected for this paper |
| 9 | Retraining with the filter in the loop | Policy unaware of the filter | L (Dawood; Choi; CBF-RL) | M | M | H (scope expansion) | M | Rejected |

Note on candidate 6: Lemma 1 caps the decay rate at 1/Δ. A state-only margin m therefore still excludes u = 0 when h < mΔ, so a decay variable cannot restore feasibility.

Rejected or deferred without a row: critic-guided safe action selection. It uses the centralized critic, which is extra information relative to all comparators, and the critic's values are uncalibrated off the training distribution.

---

## 4. Recommended method (ZR) and its guarantees

**Setting.** The unicycle, offset point p, Jacobian J(θ), barriers, box U containing 0, ranges, equal split, ZOH, exact solver and the no-fallback rule are all unchanged. Notation: v̄ = 0.5, ω̄ = 1, V_p = √(v̄² + ℓ²ω̄²).

**Lemma 1′ (one-sided ZOH; derived).** Suppose:

- u is held on [t_k, t_k+Δ];
- ḧ ≥ −M on that interval, with M ≥ 0;
- αΔ ≤ 1 and h(t_k) ≥ 0;
- ḣ(t_k) + αh(t_k) ≥ MΔ/2.

Then h(t_k+s) ≥ (1 − αs)h(t_k) ≥ 0 for all s in [0, Δ]. The proof of Lemma 1 goes through using only the lower bound on ḧ.

**Command-dependent bounds (derived).** Under a held command, s = |ṗ| = √(v² + ℓ²ω²) is constant and |p̈| = |ω|s.

- **Obstacle.** ḧ ≥ 2s² − 2(d+sΔ)|ω|s ≥ −F(ω). Here F(ω) = max over σ in [0, V_p] of [2(d+σΔ)σ|ω| − 2σ²]₊. F is convex in |ω| and F(0) = 0, so the chord gives F(ω) ≤ F(ω̄)|ω|/ω̄. Define κ_o(d) = ΔF(ω̄)/(2ω̄) = Δd²ω̄ / (4(1 − Δω̄)) for d ≤ 0.905 m. Numerically κ_o(0.35) = 0.0034.
- **Pair.** ḧ_ij ≥ −2(d + 2V_pΔ)V_p(|ω_i| + |ω_j|). The 2|ṗ_i − ṗ_j|² term is dropped because it can vanish. Define κ_r(d) = ΔV_p(d + 2V_pΔ). Numerically κ_r(0.52) = 0.0312.

**Filter (per robot, every period, same state for all robots).**

```
minimize ½‖u − u_nom,i‖²  over u = (v, ω) in U, subject to:
  for each obstacle o in sensing range, σ in {+1, −1}:
      2(p_i − c_o)ᵀ J(θ_i) u − σ κ_o(d_io) ω  ≥  −α h_io
  for each neighbour j in comm range, σ in {+1, −1}:
      2(p_i − p_j)ᵀ J(θ_i) u − σ κ_r(d_ij) ω  ≥  −(α/2) h_ij
Solve with the existing qp2d solver.
If any QP is infeasible, report INFEASIBLE (no repair). Apply u with ZOH.
```

The problem stays convex (polyhedral) and decentralized. α is fixed, and nothing is adapted online. Rows switch only by the existing range rule.

**Theorem (PROPOSED; proof sketch derived here).** Assume A1–A8 and the range rule, with all h ≥ 0 at t₀. Then for every period k:

1. u_i = 0 is feasible for every robot. Each barrier row reads 0 ≥ −αh or 0 ≥ −(α/2)h, and U contains 0.
2. Every barrier stays non-negative throughout each period. For obstacles, ḣ + αh ≥ κ_o|ω| ≥ (Δ/2)F(ω), and Lemma 1′ applies. For a pair, summing both robots' rows (A3) gives ḣ_ij + αh_ij ≥ κ_r(|ω_i| + |ω_j|) ≥ (Δ/2)M_ij. Omitted barriers cannot reach zero within the period by the range rule.
3. Body discs never overlap obstacles or each other, by the existing footprint cover. INFEASIBLE never occurs.

**Proposition (inclusion; derived).** At every state and α, the admissible sets satisfy K_C ⊆ K_B ⊆ K_ZR, strictly. The reason is m_ZR(u) ≤ m_B ≤ m_C for every u in U; for pairs, m_C = m_B + 2ΔV_p². So the ZR solution is pointwise at least as close to u_nom. This does not order closed-loop outcomes.

**Checklist:**

- **Footprint.** Same cover disc as now.
- **Information.** Identical to CBF-C: own pose, obstacles in sensing range, neighbour poses in comm range.
- **Cost.** About 2.5–3.5× more solver candidates, estimated and not timed.
- **Formation at rest.** Every configuration with h ≥ 0 is admissible at every α.
- **Formation transit.** The cap from finding 4 shrinks but remains.

**Extension ZR-0 (secondary).** Use h = |x − c|² − (R + R_o)² and h_ij = |x_i − x_j|² − (2R)² on the axle centre.

- ḣ is linear in v only. The same chord construction applies with σ ≤ v̄ (κ = 0.0025 at d = 0.30; pair κ = 0.0310).
- The theorem carries over verbatim. A state-only margin would be infeasible wherever the heading is tangent and h < δ/α, which is why ℓ > 0 was needed.
- The footprint becomes exact, and the α = 5 transit cap rises to 0.55 m/s.
- Cost: the filter loses first-order steering. It brakes or reverses and limits |ω| near the boundary. It is the certified counterpart of B-match.

---

## 5. Contribution relative to the closest prior art

**Closest competitor.** CBF-B, the Breeden-type tight state-only margin, combined with an explicit stop fallback (Chen–Singletary–Ames backup logic). With an equal split, the fallback is also safe: the neighbour's half-row covers its own motion at decay α/2.

**The exact difference.** The margin depends on the held |ω_i|, not on a supremum over U, and stays polyhedral. As a result:

- feasibility is a property of one QP rather than a switch;
- the minimal-modification property holds at every state;
- pair margins vanish when a robot is not turning.

**Strongest reviewer objection.** "A stop fallback already gives feasibility, and at α = 10 the margins are millimetres. This is a tidier proof, not a better filter."

**Result needed to answer it.** On untouched test layouts, with matched information and each arm at its own selected α:

- ZR beats CBF-B + fallback on safe arrival or transit formation error;
- COLLISION does not increase;
- or the fallback's switching visibly harms outcomes where ZR does not.

---

## 6. Failure cases and open obligations

**Derived here, needing independent checking:** Lemma 1′, both one-sided bounds, the chord step, the theorem, and the inclusion proposition.

**Existing results used under assumptions verified in code:** footprint cover, equal-split summation, the range rule, and VMAS's exact arcs (to float32).

**Unproven:**

- **Robustness to float32 tracking error.** At h ≈ 0, a ~1e-7 m error makes h slightly negative, and the u = 0 row then fails the 1e-9 tolerance. A fix exists if the simulator error vanishes at u = 0: add an input-proportional term. Otherwise feasibility is only approximate.
- **A general version for driftless systems.** It looks straightforward but is not written out.

**Performance counterexamples. None of these is a deadlock-freedom claim; ZR makes no liveness claim.**

- **Symmetric head-on encounter.** Both robots may simply stop. C's forced "repulsion" is gone.
- **Pinch between obstacles.** INFEASIBLE turns into a stop, which may become a TIMEOUT.
- **Facing an obstacle at h = 0.** Turning in place needs at least 5.6° off-axis, otherwise the robot must reverse. This is fine with v_min = −0.5, but not without reverse.
- **Narrow passages.** The barrier forbids the formation compression that 16–26% of successful filter-off runs use.
- **Formation transit.** The throttling at α ≤ 5 remains.
- **Zero nominal command.** Parking stalls are untouched.

**Needs experiments:** every effect on GOAL, safe arrival, formation, stalls and the selected α.

---

## 7. Smallest decisive plan

**Milestone 0 (analysis).** Write the proofs into the docs and get them checked by an advisor. Then add unit tests that use no study data:

- the one-sided bounds on random exact arcs, mirroring `test_hddot_bounds`;
- u = 0 is feasible at h = 0;
- the two existing constructed infeasible cases become feasible.

Abandon if any bound fails.

**Milestone 1 (development sweep).** Run on the validation layouts, after the protocol amendment is locked: 9 actors × K = 1, 2, 3 × 200 layouts × α ∈ {0.5, 1, 2, 5, 10}. Upper-bound compute is about 2–2.5 core-hours.

| Arm | Role |
|---|---|
| OFF | Baseline |
| CBF-C | Existing filter, code unchanged |
| CBF-B | Closest published margin |
| CBF-B + stop fallback | Explicitly labelled; the strongest fair comparator |
| CBF-ZR | Proposed method |
| B-match | As proposed, using C's trigger |
| ZR-0 | Optional secondary arm |
| ZR-2S | Ablation: input-dependent but two-sided margin |

C/B/ZR-2S/ZR form a 2×2 that isolates two factors: one-sided versus two-sided, and state-only versus input-dependent.

- **Gain selection.** Each arm picks its α by the same rule. For the fallback arm, the tie-break counts fallback activations instead of INFEASIBLE.
- **Pre-registered descriptive predictions:**
  - C at α ≤ 1 shows forced formation expansion or INFEASIBLE.
  - Every arm at α ≤ 2 intervenes during open-space transit.
  - ZR has zero INFEASIBLE.
- **Abandonment criterion.** At the selected gains, pooled over 5,400 pairs, ZR versus B + fallback differs by:
  - less than 1 pp in both safe arrival and GOAL;
  - less than 0.002 m in transit formation error;
  - with fallback active in fewer than 0.5% of trials.

  If so, drop the performance claim, keep the theorem as a remark, and proceed with the empirical study. Also revise or abandon if ZR loses 2 pp or more of GOAL to B + fallback because of stalls near obstacles.
- **Continuation criterion.** A gain of 2 pp or more in safe arrival or GOAL with no formation loss, or fallback activations of 1% or more at B's selected gain.

**Final test.** Use the reserved seeds unchanged: arms OFF, ZR, B + fallback, C, B-match, at their locked gains, 1,000 layouts per policy and density. Within each density, test in fixed sequence; use Bonferroni 0.05/3 across densities.

1. ZR versus OFF: the usefulness criteria.
2. ZR versus B + fallback: safe-arrival superiority, plus transit formation non-inferiority at 0.01 m, using a paired mean confidence interval (not Hodges–Lehmann).

Report all nine actors per seed.

**Confound controls.** The tolerances, the 400-period horizon, observations and actors are all unchanged. Report mean speed and time to the 0.30 m region, so a gain cannot come from slower motion.

**Required protocol amendments:**

1. Add the new arms and per-arm gain selection; document the change in role of the validation layouts.
2. Fix the formation estimand: use a transit-window mean, treat zero-length trials as missing with a sensitivity analysis, and pre-specify terminal handling.
3. Log terminal-state and per-arc barrier minima, active rows, fallback activations and speed.
4. Fix a rule for numerical tolerance at h ≈ 0.
5. Pre-register the failure-mode classifier: near-goal stall, obstacle-adjacent stall, transit throttling, infeasible, collision.
6. Reconcile the endpoints and statistics with the committed draft.

---

## 8. Candid decision

Pursue the methodological contribution, but bounded and gated: one development sweep decides it. The math and implementation are solid and cheap.

The empirical payoff is uncertain. Most of the conservatism reduction relative to the current filter comes from established theory (CBF-B). Equal-split transit throttling will likely push α* to 5–10, where ZR's advantage shrinks.

Whatever happens to ZR, adopt CBF-B as a comparator, fix the formation estimand, and add the transit diagnostics before locking. Otherwise the empirical study itself is exposed to reviewer criticism.

---

## Closing items

- **Provisional title:** Keeping Stop Admissible: Input-Proportional Sampled-Data Barrier Margins for Decentralized Safety Filtering of Frozen Multi-Robot Formation Policies
- **Research question:** Does a sampled-data CBF tightening proportional to the held command eliminate filter infeasibility and reduce the task cost of decentralized execution-time filtering on frozen MAPPO formation policies, compared with the tightest state-only margin with a stop fallback, at matched information and gains?
- **Contributions:**
  - **PROPOSED:** A polyhedral, zero-at-rest ZOH tightening for unicycle teams, with a recursive-feasibility and intersample-safety theorem that removes the per-step feasibility assumption. Its admissible set strictly contains those of existing state-only margins.
  - **PROPOSED:** Formation-compatibility conditions for decentralized sampled-data filters (at-rest gain thresholds, transit cap αh/(4d)), and an exact-footprint instantiation that only the zero-at-rest margin makes feasible.
  - **PROPOSED:** A pre-registered paired evaluation on nine frozen MAPPO actors that attributes failures to nominal stalls, filter stalls, throttling, infeasibility or collision.
- **First practical step:** Write Lemma 1′, the theorem and the inclusion proof, plus the bound and constructed-case unit tests, and draft the protocol amendment for advisor review. Nothing gets run on study layouts until the amendment is locked.
- **What would make me abandon it:** At the selected gains on the development layouts, ZR is within 1 pp of CBF-B + stop fallback on both safe arrival and GOAL, with negligible fallback use. A failed numerical check of the one-sided bound would also end it.

---

## Sources

- [Vinod et al., ICRA 2022](https://www.merl.com/publications/docs/TR2022-053.pdf)
- [Dawood et al., arXiv 2312.12861](https://arxiv.org/html/2312.12861v3)
- [Choi et al., arXiv 2505.02293](https://arxiv.org/abs/2505.02293)
- [Singletary et al., CDC 2020](https://arxiv.org/pdf/2005.06418) ([Caltech record](https://authors.library.caltech.edu/records/gwsf1-f2s38))
- [Breeden et al., L-CSS 2022](https://arxiv.org/abs/2103.03677)
- [Tan et al., ZOCBF](https://arxiv.org/html/2411.17079v2)
- [Liu et al., SACBF](https://arxiv.org/html/2511.11897v2)
- [Yang et al., ACC 2019](https://arxiv.org/pdf/1903.03692)
- [Chen et al., L-CSS 2021](https://par.nsf.gov/servlets/purl/10281485)
- [Emam et al., CDC 2019](https://arxiv.org/pdf/1909.02966)
- [Bahati et al.](https://arxiv.org/pdf/2304.08685)
- [Oruganti et al.](https://ar5iv.labs.arxiv.org/html/2309.08050)
- [Singh et al.](https://arxiv.org/html/2610.17916)
- [Kim](https://arxiv.org/html/2603.06954v1)
- [Wang et al., T-RO 2017](https://authors.library.caltech.edu/records/tshzw-g4v69)
- [Zeng et al.](https://arxiv.org/abs/2103.12375)
- [Zhou et al., BVC](https://msl.stanford.edu/bibliography/zhou_fast_2017.html)
- [Taylor et al.](https://arxiv.org/abs/2203.11470v2)
- [Reis et al.](https://arxiv.org/pdf/2003.07819)
- [Tan & Dimarogonas](https://people.kth.se/~dimos/pdfs/Xiao_Automatica_2024.pdf)
- [Grover et al.](https://arxiv.org/pdf/1911.09146)
- [Lyu et al.](https://arxiv.org/pdf/2206.09030)
- [Cosner et al.](https://ar5iv.labs.arxiv.org/html/2303.03504)
- [arXiv 2410.07409](https://arxiv.org/pdf/2410.07409)
- [CBF-RL](https://www.alphaxiv.org/abs/2510.14959)
- [Muraccioli et al.](https://arxiv.org/abs/2607.14488)
- [Gao et al. (inaccessible)](https://www.researchgate.net/publication/398910195_Safety-Critical_Finite-Time_Cooperative_Control_for_Sampled_Multi-Robot_Systems)
