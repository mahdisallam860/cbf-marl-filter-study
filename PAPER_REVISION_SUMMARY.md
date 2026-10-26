# Paper revision summary (2026-10-26)

- **Start:** 09:14:04 UTC. **End:** 09:24:52 UTC. Both are times reported by `date`.
- **Gate:** passed (HEAD was 644fed5).
- **Push:** a dry-run push succeeded and every commit was pushed to `origin/claude/elegant-hopper-v8jxlv`, so no bundle was needed.
- **PDF:** no TeX engine is available, so `paper/main.pdf` was not recompiled. It is the 644fed5 build and is now **stale**.
- `results/REPORT.md` stays untracked.

## Instructions applied

The run applied the "FINAL OVERRIDES" message, which arrived just before the main prompt, wherever the two conflict:

- phase caps of 10 / 60 / 65 / 25 / 20 min;
- the time-to-goal label "pre-declared episode-length endpoint; exploratory, post hoc joint-success analysis":
  - the difference is test arm minus filter off;
  - the mean is equal-actor-weighted;
  - the median, Wilcoxon and Hodges–Lehmann statistics are pair-weighted;
  - an estimate is reported as unavailable if an actor has no pairs;
- the chronology correction;
- the H1–H3 limits (no expansion of confirmatory claims; collision counts kept separate from the composite endpoint; no density trend claimed);
- the reworded boundary-gain implication;
- a provisional comparison table, with classifications not verified;
- the example-selection tie rules;
- no QP solver or filter call.

## Phases

| phase | status | commit |
|---|---|---|
| 0 Setup (rules appended to OVERNIGHT_RULES.md) | done | `454bc64` |
| 1 Analyses on saved data | done | `8e2cf03` |
| 2 Paper revision | done | `8bfc5e5` |
| 3 Verification (+ rounding fix) and FINISH | done | this commit (see `git log`) |

## Phase 1 outputs (`results/paper_revision/`, `scripts/paper_revision_analyses.py`)

- **(a) Time to goal** (joint-success pairs, in periods of 0.1 s):
  - Filter on vs off, equal-actor mean difference: +0.30, +0.68 and +2.09 periods at K = 1, 2, 3. The pair-weighted median is 0 at every density.
  - B-match vs off: +0.47, −0.24 and +2.62 periods.
  - Every actor has joint-success pairs at every density, so no equal-actor estimate is unavailable.
- **(b)** Per-seed locked endpoints, descriptive.
- **(c)** Validation vs final test at α = 10, exploratory.
- **(d)** Stress cells at K = 4, 5, 8, exploratory and descriptive.
- **(e)** Filter-off baseline per actor at K = 1–5 and 8.
- **(f) Example-figure selection.** The group-A median is 3.298875 cm (unrounded), and exactly one pair is closest to it:
  - selected pair: s1_late, K = 2, trial 348, clearance 3.29879 cm;
  - filter off reaches GOAL after 83 periods, with closest approach at period 21;
  - filter on: stall onset at period 78;
  - cover-disc robot 0: h = 0.00457 m², body clearance 9.5 cm.

## Phase 2 changes to `paper/main.tex`

- **Introduction:** adds the research question and the quoted Stage 2 hypotheses H1–H3, labelled as motivating hypotheses.
- **Related work:**
  - rebuilt with the Stage 2 bibliography;
  - `wang2017` merged and kept under one key;
  - the Dawood entry is now the RA-L version under the new key `dawood2025`;
  - provisional comparison table with "paired on/off" defined, plus a note that the Dawood comparison may be a training ablation (not verified);
  - one sentence cites Agarwal et al. for interval estimates and per-seed reporting.
- **System and filter:**
  - equations for the unicycle, the offset point and J (with det J = ℓ), the two barriers, the one-sided Lemma 1 (Breeden et al. Theorem 3), the implemented two-sided margins, the equal split and the per-robot QP;
  - a new analytic schematic, F0: the same 3 cm body gap gives h < 0 when heading toward the obstacle and h > 0 when heading away.
- **New section "Policy and training":**
  - observation, reward, MAPPO settings and curriculum, taken from `docs/MAPPO_POLICY.md` and `config/study.toml`;
  - the checkpoint rule;
  - the gate history: best pooled GOAL 11.0% (v1) and 28.7% (curriculum); K = 5 and 8 moved to stress;
  - the MAPPO stock-navigation check: 99.0% vs 0.0% random;
  - the baseline table.
- **Experimental design: chronology corrected.** The protocol was locked before any filter-on data with the frozen MAPPO actors existed. The 4,200-trial engineering stress tests with random and attack controllers, and the stand-in demonstration runs, came earlier. The locked document is unchanged; the correction appears in the paper (design section and disclosures).
- **Results:** adds the per-seed table, the time-to-goal table, the validation-vs-test table and the stress-cell table, each with its label.
- **Exploratory section:** adds the example paragraph and figure F6.
- **Discussion:**
  - maps H1–H3 to the evidence: confirmatory (composite endpoint, criteria b and c) vs descriptive (collision counts, stalled flag) vs exploratory (stall classes);
  - adds practical implications as suggestions, using the overrides' boundary-gain wording.
- **New sections:** Limitations (extended), Future work, Reproducibility.
- **Disclosures:** the chronology correction is added.
- **Kept:** the DRAFT header, the AI statement, all disclosures and the verbatim boundary sentence typeset as $\alpha^\star\Delta=1$.
- **Length:** about 4,600 words, 7 tables and 6 figures, roughly 11–13 pages (not compiled).
- **Numbers:** every number comes from `paper/numbers.tex`, which now has 250 macros; the anchor check passes.

## Verification (Phase 3)

- **Method:** a fresh subagent recomputed every new number from the per-trial cells, per-step logs and layouts without reading any repo script. Record: `results/paper_revision/independent_verification.md`.
- **Numbers:** all match except one rounding artifact. The baseline table cell s0_late K = 3 showed COLLISION as 57 when it is exactly 57.5%. It was fixed in `scripts/paper_numbers.py` (exact decimal half-even rounding) and now reads 58; no other macro changed.
- **Anchor check:** passes.
- **Citations:** all 21 cite keys resolve, the Stage 2 entries match the prompt exactly, and Breeden et al. plus the untouched entries are byte-identical to 644fed5.
- **Example tie:** two group-A pairs are equally close to the median, since the even-count median is the midpoint of exactly these two. The stated actor-index tie-break selects s1_late.
- **Tests:** no test suite was run, because no code under `cbf_study/` changed (recorded as required).

## Judgement calls

1. **Overrides message:** applied as described above.
2. **Optional references left out.** No primary source was reachable to verify metadata (DOI, Crossref, arXiv and PubMed all returned no response), so the MAPPO, VMAS and Newcombe references are not cited. Newcombe method 10 is named in the text without a citation.
3. **Comparison table:**
   - Vinod row: "not stated here" for both paired on/off and outcomes. The existing paper text states neither.
   - Dawood row: "Yes (Stage 2); see note".
   - The whole table is marked provisional and not verified.
4. **H3 evidence:** the share of filter-on non-GOAL trials that are INFEASIBLE or stalled-flag TIMEOUTs is 99.8%, with 25 still moving. It is computed from the per-trial cells and labelled descriptive (stalled flag pre-declared); the class split is exploratory.
5. **F0 second panel:** changed from "tangent" to "heading away", because at a 3 cm gap the tangent heading also gives h < 0 and would not show the sign change.
6. **Stale PDF:** `paper/main.pdf` was not deleted (no deletions allowed) and not regenerated (no TeX).
7. **Policy text:** design and config values (reward weights, MAPPO settings, budgets) are written in the text as configuration constants. Only empirical numbers go through macros.

## Skipped

- PDF compilation: no TeX engine.
- The MAPPO, VMAS and Newcombe references: metadata unverifiable here.
- Verification of the comparison-table classifications: papers not accessible.

## Titles to add

These entries have no title in the existing paper or in `docs/methodology/`, and were left unchanged:

- tan_zocbf (Tan, Daş, Ames, Burdick, ZOCBF)
- liu_sacbf (Liu, Xiao, Belta, SACBF)
- vandenberg2011 (ORCA)
- chen2021 (Chen, Singletary, Ames)
- emam2019 (Emam et al.)
- kim2026 (Kim)
- yang2019 (Yang, Belta, Tron)

## Decisions that need me / my advisors

1. **Paper:** framing, title and authorship, and whether the expanded draft is the right base for the later six-page IEEE version.
2. **Margins and protocol:** advisor review of the author-chosen margins and protocol; no advisor agreement exists.
3. **Comparison table:** verify each classification, in particular Dawood "paired" vs training ablation and the Vinod columns.
4. **References:** add the missing titles listed above. Decide whether to add the MAPPO, VMAS and Newcombe references once their metadata can be checked.
5. **PDF:** recompile `paper/main.pdf` and check the page count (target 10–12) and table widths.
6. **Seed-reservation test:** whether to update the failing test (from the overnight run) to exempt the authorized final-test outputs.
