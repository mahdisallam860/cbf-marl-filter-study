# Review round 1: editorial synthesis and change log (2026-10-27)

Simulated five-seat review (academic-research-skills `academic-paper-reviewer`, full mode) of `paper/main.tex` at
commit 7ff38ef. Each seat ran as a separate agent with no access to the other reports and read the paper, the locked
protocol and the committed results; none modified the repository. Execution notes: all five seats ran on the same model
family, so their errors are not independent; the skill's two-call sprint-contract machinery and checker scripts were not
run; the reviewer configuration cards were not shown to the author before the review.

| Seat | Recommendation | Findings |
|---|---|---|
| Journal-fit (EIC) | Major (RA-L); poor fit for L-CSS | 26 (10 major) |
| R1 methodology/statistics | Minor, borderline major | 15 (4 major) |
| R2 CBF/sampled-data control | Major | 18 |
| R3 MARL evaluation practice | Major | 23 |
| Devil's advocate | no CRITICAL; 10 major | 16 |

**Decision: Major revision (interpretation and definitions).** Every seat confirmed the confirmatory numbers and the
verdict; no finding challenges a number. The problems are what the text claims and how terms are defined.

## Consensus items (two or more seats) and the change made

| Issue | Seats | Change (both manuscripts) |
|---|---|---|
| Infeasibility attributed to αΔ = 1; the infeasible states sit at the stand-off, which exists at every gain, and the sweep had 1,251 INFEASIBLE at α = 0.5 | R2-1/2, EIC-1, DA-7 | Abstract, contribution (iii), Infeasibility paragraph and "Gain at the lemma boundary" rewritten; float32 sign flips (88/433), rows violated at u = 0 (1–3) and the not-identified conflict set reported; αΔ = 1 attribution withdrawn |
| Stalled flag marks ~100% of TIMEOUTs in both arms, so it cannot test H3 | R1-2, R3-4, DA-6 | H3 paragraph rewritten with the off-arm shares (99.9 / 99.9 / 100.0%); flag parameters defined in the introduction |
| "Filter-induced stall" and "in front of obstacles" are causal/spatial claims | EIC-2/3, R3-13, DA-8 | Renamed "intervention-active stall" in text, Table 9 and Fig. 5 legend; repository name noted; abstract rewritten; 5,699 stalls that replaced a collision stated |
| K = 1: GOAL estimate inside the margin; shortfall concentrated in seed 1 | R1-1/3, R3-2, EIC-6, DA-4 | Results and per-seed paragraphs, abstract and conclusion state it, with per-seed macros and "5 of 9 actors inside both margins at K = 1" |
| Margins unjustified | R1-13, R3-3, EIC-5, DA-9 | Stated as the author's judgement; Table 3 bounds identified as the smallest passing margins; safety/completion trade-off stated descriptively |
| Weak policy family: failed competence gate, densities chosen after filter-off results, K = 1, 3 outside the final stage | R3-1/6, EIC-4, DA-10 | Section 4, abstract, limitations, disclosures and conclusion |
| 0.05/3 split and the (a) decision rule | R1-4/12, R3-15 | Bonferroni over density claims, IUT within density, McNemar decides (a) |
| Newcombe label: the protocol's own text describes method 8 | R1-5, EIC-20 | Design paragraph and disclosure rewritten |
| Predictions over-claimed; the methodology report's margin warning omitted | R1-10, R3-11, EIC-11, DA-2 | Removed from contributions; locked D4 prediction cited; 13× margin estimate and the unaddressed advisor-review recommendation disclosed |
| H1 support is by construction (terminated infeasible trials; uncertified brake also zero) | R2-12, R1-15, EIC-14, DA-15, R3-17 | H1 paragraph rewritten |
| B-match cannot be infeasible by construction; clipped ω; physical meaning of v = 0 | R2-11, R1-11, R3-10, EIC-16 | Section 3, results, Table 4 caption, implication (iv) |
| AUC is in-sample, post hoc and partly mechanical | R1-15, R3-9, EIC-15, DA-12 | Clearance paragraph and implication (iii) |
| Undefined: stand-off, slack (sign and order), diagnostic, intervention, closing-speed cap, d and c, ranges, A1–A9, layout/density, stall classes, INFEASIBLE timing, GOAL timing, r, "held-out", "first configuration", "Stage 2", ZR | R2-3…15, R3-5/7/12/18–22, EIC-7/8/17–23, DA-13/14, R1-8/9 | Definitions added in Sections 3–5 and 7; formation estimand renamed Δe to free d |
| Cover-disc conservatism understated ("a few centimetres") | R2-7, R3-8 | 2ℓ = 10 cm bound and the 0.6–10.6 cm clearance band a stopped robot needs; connected to group A in the discussion |
| Label errors: stress cells pre-specified; time-to-goal label; Wilcoxon on tied data | R1-6/7 | Relabelled; Wilcoxon caveat in text and caption; Section 6 renamed "Results" |
| Missing implication: fallback and deadlock recovery | R3-16 | New implication (i) |

## Not done in this round (and why)

- Length and venue fit (EIC-9): deferred to the conference manuscript step.
- Positioning against single-agent shielding (EIC-10): needs a verified literature search.
- Identifying the conflicting row set or a Farkas certificate for each infeasible QP (R2-2): new analysis; the paper now
  says the audit did not identify it.
- Per-paper verification of the sampled-data references (R2-16, DA-16): the blanket claim was narrowed instead.
- Actor-level rank correlation of clearance and cost (R3-9), relative margin sizes (R1-13), the checkpoint spacing
  (20 iterations saved vs `checkpoint_every = 10`, R3-20): not done; the last should be checked in the training logs.
- Table 4 keeps method 10 (the task specification); method 8 values are in the CSV (R1-5).

New citation: Reis, Aguiar, Tabuada, IEEE L-CSS, doi:10.1109/LCSYS.2020.3004797 (verified on arXiv:2003.07819 and the
authors' lab page; volume and pages not verified, so omitted).

All new numbers come from `scripts/paper_numbers.py` (`review_round1_macros`, 56 new macros; no existing macro changed;
`numbers.tex` regenerates byte-identically). Only figure F7 was regenerated (legend label).
