# Devil's Advocate report (seat 5 of 5)

Manuscript: "An Execution-Time CBF-QP Safety Filter on Frozen MAPPO Formation Policies: A Pre-Specified Paired Evaluation in Simulation"
Line numbers refer to `paper/main.tex`. Quotes are from main.tex; where a quote contains a macro, its rendered value is given in brackets.
Evidence beyond the paper: `results/final_test/FINAL_TEST.md`, `results/final_test/INFEASIBILITY_AUDIT.md`, `results/final_test/MECHANISM_EXPLORATORY.md`, `docs/CBF_DESIGN.md`, `docs/PREFILTER_VALIDATION_PROTOCOL_LOCKED_2026-10-25.md`, `docs/methodology/CBF_MARL_methodology_report_2026-10-25.md`, `config/study.toml`.

---

## 1. Strongest counter-argument

The headline, "not shown useful at any density", judges one boundary instantiation against acceptability margins the author chose without review. The framing lets readers take it as a verdict on execution-time CBF filtering of MAPPO policies in general. Four facts from the repository limit how far it generalises.

(1) **The instantiation sits at every extreme.** α\* is both the top of the grid and the lemma's αΔ ≤ 1 cap. Filter-on GOAL rose monotonically across the grid (0 → 2,224), so the selection rule stopped at a constraint, not at an optimum. The barrier is a cover disc that widens the body footprint laterally by ℓ = 5 cm, and the lost completions came from unfiltered passes with a median clearance of 3.3 cm. The margin uses a two-sided bound. The author's own pre-sweep methodology report estimated it at about 13× a tight one-sided margin (obstacle, d = 0.35 m) and said it "should become an advisor decision". There is no fallback.

(2) **A trivial brake on the same rows matched it.** B-match had zero collisions and zero INFEASIBLE outcomes, and it reached GOAL within 1.0–3.2 pp of the filter. The finding that matters for a decision is that the QP projection adds little once the CBF rows are used as a trigger. Neither the abstract nor the conclusion says this.

(3) **The K = 1 verdict rests on one seed.** Seeds 0 and 2 have GOAL losses of 2.9 and 4.0 pp and formation increases of about 0.005 m, both inside the margins. Seed 1 has 7.5 pp and 0.025 m, and two of its actors come from a crash-resumed run.

(4) **The nominal policies are weak.** They failed their own pre-registered competence gate (best 28.7% against 50%). They also pass within centimetres of obstacles despite a proximity penalty that starts at 15 cm. A projection filter on such policies would be expected to turn grazes into stalls.

The conclusion the data support is narrower: for these weak, obstacle-grazing policies, this conservative filter at its boundary gain removes 5–44 pp of collision-or-infeasible outcomes at a cost of 5–11 pp of completions, and a brake on the same rows does about as well.

---

## 2. Issue list

No issue is rated CRITICAL. The verdict sentences are scoped ("not shown useful under the pre-set criteria", "these nine actors"), so no headline claim is false as worded. The problems below concern framing and definitions, and they shape how the verdict will be read.

### DA-1 — MAJOR — overgeneralisation
- **Lines:** 28–30 (abstract), 74–75 (research question), title.
- **Quote:** "We evaluate whether an execution-time, decentralised control-barrier-function quadratic-programme (CBF-QP) safety filter, placed between frozen multi-agent PPO (MAPPO) formation policies"
- **Problem:** The abstract and the research question use the indefinite article. The scope sentence ("these nine actors in this simulated scenario") limits actors and scenario, but not the filter. The verdict actually concerns a specific design:
  - a cover-disc barrier (ℓ = 0.05 m);
  - command-independent, two-sided margins (lines 172–173 concede that "only the lower bound is needed");
  - the largest gain the lemma admits, which is also the grid maximum;
  - equal split;
  - no fallback.

  `config/study.toml` marks every one of these values, including ℓ and the α grid, `approval = "pending"`.
- **Fix:** Add to the abstract after the α\* sentence: "The instantiation is conservative by construction: a cover-disc barrier (offset 5 cm), command-independent two-sided margins, the largest gain the sampled-data lemma admits (α⋆Δ = 1, also the top of the grid) and no fallback; the verdict concerns this instantiation." Change line 74 to "Does this execution-time CBF-QP filter …".

### DA-2 — MAJOR — cherry-picking
- **Lines:** 70; 669–674.
- **Quote:** "documented analytic predictions that the development data matched"
- **Problem:** The paper reports only the predictions from the methodology report that came true:
  - α\* ∈ [5, 10], which covers 2 of the 5 grid points, including the cap, on a monotone GOAL curve, so it is a weak test;
  - conflict between small gains and holding the formation.

  It leaves out the same report's predictions that bear directly on the verdict:
  - finding 1: the implemented obstacle margin is 0.0454 against 0.0034 for a tight one-sided margin (13×), and "A reviewer can attribute GOAL loss to this avoidable conservatism. This should become an advisor decision";
  - finding 5: "The cover-disc barrier, not only the margin, will cost task completion."

  The margin was then locked unchanged without that review (line 723).
- **Fix:** Remove "that the development data matched" from contribution (iii). Add to Predictions or Limitations: "The same report estimated the implemented two-sided obstacle margin at about 13 times a tight one-sided margin (d = 0.35 m), predicted that the cover-disc barrier would cost completions, and recommended advisor review of the margin; the margin was locked unchanged without that review. How much of the GOAL loss is due to margin conservatism, cover-disc geometry or equal split was not tested."

### DA-3 — MAJOR — interpretation (omission in headline text)
- **Lines:** 33–34 (abstract), 741–744 (conclusion).
- **Quote:** "the filter on at the gain $\alpha^\star=\AlphaStar$ selected on validation layouts, and a brake comparator."
- **Problem:** The abstract names B-match but gives none of its results. B-match is an uncertified brake on the same rows:
  - 0 collisions and 0 INFEASIBLE at every density;
  - GOAL 48.8 / 53.5 / 19.3% against the filter's 49.9 / 56.8 / 21.5%;
  - at K = 3 the filter had 3.91 pp more terminal failures in exchange for 2.18 pp more GOAL.

  For a practitioner this is the most actionable comparison in the paper, and it bears on what the QP adds. It appears only in Section 6 and in suggestion (iii).
- **Fix:** Abstract: "A brake applied whenever the clipped command violated the same rows (B-match; not certified) had no collision or infeasible outcomes and reached GOAL 1.0–3.2 points less often than the filter (pre-declared secondary, descriptive)." Add the same clause to the conclusion.

### DA-4 — MAJOR — interpretation
- **Lines:** 412–415; 36–38.
- **Quote:** "Per-actor GOAL losses vary widely (Fig.~\ref{fig:f3})."
- **Problem:** At K = 1 the pooled shortfall on both (b) and (c) comes from seed 1 (Table 5 and FINAL_TEST.md):

  | | Seed 0 | Seed 2 | Seed 1 |
  |---|---|---|---|
  | GOAL loss [pp], 95% interval | 2.9 [2.3, 3.5] | 4.0 [3.3, 4.8] | 7.5 |
  | Formation increase [m] | 0.0047 | 0.0052 | 0.0252 |

  - Seeds 0 and 2 sit inside both margins.
  - Seed 1's s1_middle and s1_late come from the resumed run. s1_early has an anomalous filter-off K = 1 GOAL rate (20.4%) and a formation increase of 0.0417 m.
  - Five of nine actors (all of seed 0, s2_early, s2_middle) have both 95% upper limits inside the margins at K = 1.
  - The pooled (b) 95% interval at K = 1, [4.35, 5.27], straddles the margin. At K = 2 and 3 it lies wholly above it, and every seed exceeds both margins.

  "Not shown useful at any density" therefore treats a near-miss driven by one seed (K = 1) the same as clear excess losses (K = 2, 3). The pre-specified equal weighting is correct. The point is reporting, not re-analysis.
- **Fix:** After line 415: "At K = 1 the pooled shortfall comes from seed 1 (GOAL loss 7.5 pp, formation increase 0.0252 m; two of its actors from the resumed run); seeds 0 and 2 had 2.9 and 4.0 pp and 0.0047 and 0.0052 m, inside both margins (descriptive). At K = 2, 3 every seed exceeded both margins." In the abstract: "…not met; at K = 1 the GOAL bound (5.31 pp) narrowly exceeded the margin."

### DA-5 — MAJOR — definition
- **Lines:** 65; 49–50; 648–649.
- **Quote:** "failures fell at every density, but non-inferiority in goal completion and formation error was not established"
- **Problem:** "Failure" has two meanings in the paper.
  - At line 65 it means COLLISION-or-INFEASIBLE.
  - At line 648 ("Of the … filter-on trials that did not reach GOAL") it means non-GOAL.
  - Line 50 ("reduces failures without an unacceptable loss of task completion") is ambiguous.

  Non-GOAL outcomes rose at every density: 45.3 → 50.1%, 33.4 → 43.2%, 68.0 → 78.5%. A reader taking "failures fell" in the task sense will be misled.
- **Fix:** Line 65: "collision-or-infeasible outcomes fell at every density while non-GOAL outcomes rose (45.3→50.1%, 33.4→43.2%, 68.0→78.5%)". Use "safety failure" for the composite and "non-GOAL" for task failure throughout.

### DA-6 — MAJOR — definition/logic
- **Lines:** 648–652; 78–82.
- **Quote:** "\HthreeShareAll\% [99.8%] ended INFEASIBLE or as TIMEOUT with the protocol's stalled flag"
- **Problem:** The stalled flag is centroid progress below 0.05 m over the last 50 of 400 periods. It marks almost every TIMEOUT in both arms (FINAL_TEST.md):

  | Arm | TIMEOUT / stalled (%), K = 1 | K = 2 | K = 3 |
  |---|---|---|---|
  | Filter off | 39.9 / 39.9 | 16.9 / 16.8 | 20.0 / 20.0 |
  | Filter on | 50.1 / 50.0 | 42.4 / 42.3 | 74.6 / 74.5 |

  The horizon is about 5× the median filter-off completion time of 74 periods (methodology report). A "slow but moving" TIMEOUT is therefore nearly impossible by construction, and the 99.8% figure cannot separate deadlock from slow progress. The same statement holds without the filter, so it says nothing filter-specific. It also counts 5,670 near-goal parking stalls, 5,362 of them on layouts where the unfiltered run also timed out.
- **Fix:** Replace with: "The stalled flag marks almost every TIMEOUT in both arms (filter off: 39.9% TIMEOUT, 39.9% stalled at K = 1), so it cannot distinguish deadlock from slow progress at this horizon, and H3 is not tested by it. Of the filter-on TIMEOUTs, the exploratory classes in Section 7 label 61% as row-blocked and 38% as near-goal parking, the latter mostly on layouts where the unfiltered run also timed out."

### DA-7 — MAJOR — interpretation
- **Lines:** 41–42 (abstract); 602–609; 664–667; 684–687.
- **Quote:** "within \InfHdiffMax\,m$^2$ of its stand-off, consistent with numerical sensitivity at $\alpha\Delta=1$"
- **Problem:** The abstract ties infeasibility to αΔ = 1. Three pieces of evidence point to a broader cause.
  - **Infeasibility is not specific to αΔ = 1.** The development sweep recorded 1,251 INFEASIBLE trials at α = 0.5 (αΔ = 0.05), against 129 at α = 10. The stress sweep recorded 231 of 600 at α = 0.5 (CBF_DESIGN.md §8).
  - **The design itself makes the stop inadmissible below the stand-off.** The methodology report says: "u = 0 violates a row whenever h < m/α. A9 is therefore a real assumption." That holds at every gain. The paper's own Remark (ZR) exists to fix exactly this.
  - **The zero-command probe does not identify the cause.** The audit looks at the zero command only. QP infeasibility requires a conflict among 2–5 enabled rows and the box, and the audit did not characterise that conflict.

  The audit's strongest numerical evidence is left out: float32 recomputation flips the sign of the zero-command slack in 88 of 433 cases.
- **Fix:** In the abstract, replace the clause with: "consistent with the implemented margin making the stop inadmissible just below the stand-off (a property at every gain), combined with rounding there: float32 recomputation flipped the sign of the zero-command slack in 88 of 433 cases." At line 667, add: "Infeasibility also occurred at small gains (1,251 trials at α = 0.5 in the development sweep), by a different route (Section 8, Predictions)."

### DA-8 — MAJOR — definition (causal label)
- **Lines:** 509–511; 576–577; 39–40; 68–69.
- **Quote:** "were classified as filter-induced stalls (the clipped nominal command violated an enabled barrier row in at least half of the last 50 periods)"
- **Problem:** The rule records that the policy's command and the rows conflicted, nothing more. The label "filter-induced" asserts a cause, which contradicts the Section 7 disclaimer ("support no causal … claim"). In 5,699 of the 9,211 cases the unfiltered run collided on the same layout. Those stalls replaced collisions, so they are the filter doing its job, not a cost. Contribution (iii) adds causal wording of its own: "where the cost arose".
- **Fix:** Rename the class "row-blocked stall" throughout, including the table and figure legends. At line 511 add "(5,699 of them on layouts where the unfiltered run collided)". Line 68: "analyses of where the lost completions occurred".

### DA-9 — MAJOR — definition
- **Lines:** 49–50; 36–38; 271–274.
- **Quote:** "reduces failures without an unacceptable loss of task completion or formation quality"
- **Problem:** "Unacceptable" is set operationally by the 5 pp and 0.01 m margins. The locked protocol calls these "analysis decisions" (line 52 of the protocol) and gives no task-based derivation, and no advisor reviewed them. The 0.01 m margin is a tenth of the formation tolerance in the GOAL definition (0.10 m). The rule is conjunctive and has no exchange rate between safety and completion. At K = 3 it rates a 44.1 pp fall in collision-or-infeasible outcomes against a 10.6 pp GOAL loss as "not useful". Many readers would weigh that trade differently. This is disclosed in Section 12, but the abstract presents the criteria as neutral.
- **Fix:** Intro: "…without a loss larger than pre-set margins (5 pp GOAL, 0.01 m formation error) chosen by the author without advisor review; other weightings of safety against completion could reach a different judgement." Add "(author-set, not advisor-reviewed)" after "criteria fixed before the test" in the abstract.

### DA-10 — MAJOR — overgeneralisation/interpretation
- **Lines:** 30–31 (abstract); 239–241.
- **Quote:** "Nine frozen actors (three training seeds, three checkpoints each) were evaluated on reserved, previously unused layouts"
- **Problem:** The abstract does not give the competence of the policies. Their filter-off GOAL was 54.7 / 66.6 / 32.0%, they collided in 48% of trials at K = 3, and they failed the study's own pre-registered gate (best 28.7% against 50%). A filter's cost depends heavily on how the nominal policy behaves near obstacles. Group A's median unfiltered clearance was 3.3 cm, even though training penalised clearance below 15 cm. Without this context the negative verdict reads as if it applied to competent policies.
- **Fix:** Abstract: "Nine frozen actors (…), whose filter-off GOAL rates were 32–67% and which did not meet the study's original 50% competence gate, were evaluated…".

### DA-11 — MINOR — interpretation
- **Lines:** 639–643.
- **Quote:** "These point estimates exceed the 0.01\,m margin, and criterion (c), which tests non-inferiority, was not met at any density"
- **Problem:** The pre-declared secondary "formation error over the full trajectory in pairs where both arms reached GOAL" (FINAL_TEST.md) shows increases of only 0.0008, 0.0031 and 0.0043 m. Most of the common-prefix increase therefore comes from pairs where the filtered run deviated or stalled, the same events behind (b). Criteria (b) and (c) are not independent evidence of two separate costs. The paper defers these numbers to a file.
- **Fix:** Add: "In pairs where both arms reached GOAL, the full-trajectory formation error rose by 0.0008, 0.0031 and 0.0043 m (pre-declared secondary; conditioned on joint success), so most of the common-prefix increase arises in pairs whose trajectories diverged."

### DA-12 — MINOR — cherry-picking/interpretation
- **Lines:** 595–600; 688–690.
- **Quote:** "\ThrMatchObstFive\% [43.1%] did so while keeping a sampled minimum obstacle clearance of at least 5\,cm, against a filter-on GOAL rate of \ThrOnGoal\% [42.7%]"
- **Problem:** The 5 cm cut equals ℓ, was chosen post hoc, and is the only threshold quoted in the text. The disclaimer follows it, but setting 43.1% beside 42.7% invites the "filter ≈ 5 cm buffer" reading. The AUC of 0.96 is close to mechanical: small clearance is what activates the cover-disc rows. Suggestion (ii) nonetheless turns it into deployment advice.
- **Fix:** Line 596: add "(the cut equals ℓ and was chosen post hoc; Fig. 6 shows all thresholds)". Suggestion (ii): "…the separation (AUC 0.96) is expected by construction, since the rows activate at small clearance; it is not a validated predictor."

### DA-13 — MINOR — definition
- **Line:** 149.
- **Quote:** "an infeasible QP ends the trial as INFEASIBLE before any motion"
- **Problem:** This reads as if INFEASIBLE trials never moved. Lines 288–289 make clear that only the attempted period is prevented.
- **Fix:** "an infeasible QP ends the trial as INFEASIBLE at the attempted period, before that period's motion".

### DA-14 — MINOR — definition
- **Line:** 608.
- **Quote:** "\InfNear{} of \InfRobots{} met a pre-declared near-boundary diagnostic"
- **Problem:** Elsewhere "pre-declared" means fixed in the locked protocol. This diagnostic was fixed before a post hoc audit (INFEASIBILITY_AUDIT.md: "fixed before computing").
- **Fix:** "met a near-boundary diagnostic fixed before the (post hoc) audit computation".

### DA-15 — MINOR — interpretation
- **Line:** 634.
- **Quote:** "\emph{H1 (collisions fall).} Supported for these actors: with the filter on there were \CollOnone, \CollOntwo{} and \CollOnthree{} collisions"
- **Problem:** The 433 INFEASIBLE trials stopped before their outcome was known. The uncertified B-match also had 0 collisions, so in this scenario avoiding collisions did not need the QP or its certificate.
- **Fix:** Append: "6, 75 and 352 trials were terminated as INFEASIBLE, so their later course is unknown; the uncertified B-match also had no collisions."

### DA-16 — MINOR — definition (related work; verify)
- **Lines:** 104–106.
- **Quote:** "these margins are input-independent bounds over the input set and do not depend on the applied command"
- **Problem:** This is a blanket claim over [13–16]. The title of [15] is "…with state and input dependent safety constraints". I have not checked the text. If any of the four uses command-dependent margins, the claim is wrong, and so is the contrast drawn in Remark (ZR).
- **Fix:** Check each paper. If any doubt remains, write "margins of the form used here are input-independent bounds…" and drop the blanket claim.

---

## 3. Ignored alternative explanations

1. **Cover-disc geometry against layout passages.** Obstacles may be placed with surface gaps as small as 0.10 m (`obst_gap`). The cover disc widens each side of the body by ℓ, so gaps between about 0.20 m and 0.30 m are physically passable but blocked for the barrier point. That could account for row-blocked stalls on layouts the unfiltered policy completed. It can be tested from the saved layouts and poses without new trials.
2. **Known undesirable equilibria of min-norm CBF-QPs.** Stalls in front of obstacles while the nominal command points into them are a documented property of CBF-QP safety filters. The stalls may therefore be generic to projection filters without deadlock resolution rather than specific to these actors. The literature is not cited: Reis, Aguiar and Tabuada, IEEE L-CSS 2021, on CBF-QPs introducing undesirable stable equilibria; Grover et al. on multi-robot deadlock. Both are from my own knowledge, so the author should verify them.
3. **Distribution shift at the primary densities.** The final curriculum stage trained on K ∈ {2, 5, 8}. K = 1 and 3 were seen only in earlier stages. Filter-off GOAL is lower at K = 1 (54.7%) than at K = 2 (66.6%), with 39.9% TIMEOUT at K = 1. Policy behaviour off the training mixture, including near-goal parking, may drive part of the per-density pattern.
4. **Seed-specific training artefacts.** The K = 1 shortfall comes from seed 1, two of whose actors come from a run resumed without its environment or global random state (see DA-4).
5. **Margin conservatism against geometry.** The paper never separates the contributions of the two-sided margin, the cover disc and the equal split. At α = 10 the obstacle stand-off from the margin is only 0.65 cm, while the lateral inflation from the cover disc is 5 cm. The geometry may matter more than the margin. The paper cannot tell either way, and the methodology report flagged both.
6. **Policy proximity behaviour.** Training penalised clearance below 15 cm, yet the policies learned to graze obstacles (group A median 3.3 cm). The "cost" may mostly reflect a nominal policy whose successful paths lie inside the filter's forbidden band.
7. **Duration effects in criterion (c).** In pairs where both arms time out, the common prefix is the full 400 periods, including long stationary stretches in distorted configurations. s1_early at K = 1 has a GOAL loss of only 0.4 pp but a formation increase of 0.0417 m.

---

## 4. Observations that are not defects

- **No fallback does not drive the verdict.** Only 9 of the 2,540 lost completions ended INFEASIBLE. A brake fallback would change (a) slightly and leave (b) essentially unchanged.
- **α\* at the grid edge favours the filter on (b).** The rule picked the highest-GOAL gain, and lower gains did worse. The boundary choice is the filter's best case for completion. Its cost is infeasibility (DA-7), not a biased verdict.
- **Margin conservatism is small at α = 10.** The stand-off from the margin is 0.65 cm for obstacles and 2.3 cm for pairs. The "margin choice" line of attack is weaker than the "cover-disc geometry" line. DA-2 asks for disclosure, not a claim that the margin explains the loss.
- **The composite endpoint (a) counts filter-caused INFEASIBLE outcomes against the filter.** This is conservative and was pre-specified. (a) passes overwhelmingly regardless.
- **B-match shows negative cover barriers (min h_obst −0.025 m²) with zero collisions.** This is consistent with h < 0 not implying contact (Fig. 1), not a contradiction.
- **Method 8 against method 10.** The difference is below 0.001 pp and is disclosed. Table 4 uses method 10 and Table 3 uses method 8, which is also disclosed.
- **Validation against final test (Table 7).** The dev sweep already showed the K = 2, 3 failures. The protocol requires reporting a failing α\*, so this is consistency, not a problem.
- **Net loss equals the criterion (b) estimate.** This is an arithmetic identity, correctly stated.
- **Equal-actor weighting.** With equal allocation it equals pair pooling.
- **A5 exception.** The single trial is kept and disclosed.
- **Joint-success time to goal.** The selection caveat is stated correctly at line 436.
- **"Not shown useful" against "shown not useful".** Lines 642–643 correctly separate failed non-inferiority from demonstrated inferiority.
