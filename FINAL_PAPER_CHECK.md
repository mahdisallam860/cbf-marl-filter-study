# Final paper check (2026-10-27)

Starting commit: `7c46bb7` (branch `claude/elegant-hopper-v8jxlv`). Files changed: `paper/main.tex`, `paper/main.pdf`,
`paper/ieee/main_ieee.tex`, `paper/ieee/main_ieee.pdf`, this file. The same 16 replacements were applied to both
manuscripts (whitespace-tolerant match, each found exactly once in each file).

## Corrections

1. **Lower-gain statement.** Implication (ii) no longer says lower gains were infeasible more often. It now says GOAL
   increased across the grid, infeasibility was non-monotonic, lowering α from 10 to 5 reduced both GOAL
   (\SelGoalten → \SelGoalfive) and INFEASIBLE (\SelInften → \SelInffive), and the validation results do not show that
   a lower gain improves overall usefulness. Gain-selection rule and α* unchanged.
2. **Stalled flag.** Checked against `cbf_study/protocol.py` (`progress = cen_dist[end−50] − cen_dist[end]`,
   `stalled = TIMEOUT & progress < 0.05`): it measures the fall in centroid-to-goal distance over the last 50 periods.
   Defined that way in the introduction and the stall-class definitions, with the note that sideways, circling or
   receding motion also satisfies it. "Almost every TIMEOUT is stationary" replaced by low terminal progress toward
   the goal; H3 text states the flag cannot distinguish parking from obstacle- or filter-held teams and does not
   identify filter-specific deadlock; "stall" defined as low terminal progress; classes called operational labels.
   Thresholds, rules and saved category names unchanged. "May reflect" replaces "are consistent with" for the
   undesirable-equilibria remark.
3. **Rounding.** The float32 sign change (88/433) is now described as sensitivity of that row's evaluation only; the
   text states it does not show whole-QP feasibility, zero-command admissibility for other rows, or a different trial
   outcome, and that the conflicting set and the effect on feasibility and later outcomes are unresolved. The
   "not specific to αΔ = 1" conclusion is kept, with "a numerical contribution of the boundary gain is not excluded".
4. **Zero collisions.** "Confirms the implementation" replaced: consistent with the conditional guarantee, numerical
   evidence on the evaluated trajectories, not a certification of the solver, of every assumption or of
   floating-point safety. The proposition and A1–A9 are unchanged.
5. **Competence gate.** Abstract, limitations and conclusion now state the gate was pooled over K = 2, 5, 8; the
   choice of K = 1, 2, 3 after seeing filter-off results is kept; K = 1, 3 described as absent from the final
   curriculum stage but encountered in earlier stages.
6. **Zero-prefix pairs.** Design section now says the formation difference is undefined and excluded, each arm keeps
   its recorded outcome (an infeasible arm counts as failure and not GOAL), and no pair is assigned a GOAL loss by
   the exclusion. No calculation changed (there were no such pairs in the final test).
7. **Reis et al.** Entry completed with vol. 5, no. 2, 2021 and the DOI. Page numbers **not added**: the UCLA CyPhy
   Lab publications page (accessible) lists IEEE L-CSS 5(2), 2021 and the DOI but no pages; the Universidade do
   Porto SIGARRA record could not be retrieved (client error). Pages 731–736 remain unverified.
8. **Consistency.** Both manuscripts searched for "stationary", "infeasible more often", "confirms the implementation",
   "own competence gate", "occur only in earlier", "outside the final", "remain failures": no remaining hits. The
   method-8/10 disclosure, pre-declared vs exploratory labels, the K = 1 estimate/bound distinction, equal actor
   weights, all disclosures, the AI-assistance statement and "not shown useful under the pre-set criteria" are kept.

## Verification

- `scripts/paper_numbers.py`: anchor check passed, 431 macros; `paper/numbers.tex` SHA-256
  `03fb5007621ea4515214a4dcf94870edd6aa6eb545e0cb61eb7489127f57f478`, identical to the starting version.
- Protected files: no change under `results/`, `cbf_study/`, `config/`, `scripts/` or the locked protocol
  (`git diff --name-only 7c46bb7` on those paths is empty). No figure regenerated. `results/REPORT.md` not present
  in this clone.
- Compilation (pdfLaTeX via latexmk): both manuscripts 0 errors, 0 undefined references, 0 overfull boxes.
  `main.pdf` 22 pages, `main_ieee.pdf` 13 pages (unchanged counts). Pages 9–12 of the IEEE version (mechanism,
  discussion, implications, limitations) rendered and inspected: no clipping, tables and figures readable.

## Unresolved

- Reis et al. page numbers (see item 7).
- Items deferred in `docs/review/round1/EDITORIAL_SYNTHESIS.md` ("Not done in this round") remain open.
