# Independent verification of the revised draft (paper revision, phase 3)

A fresh subagent was given only `paper/main.tex`, `paper/numbers.tex`, committed result files and the
expected bibliography from the prompt. It did not read or run `scripts/*.py`, ran no simulation, and
changed nothing. Its own scripts recomputed every new macro from the most primary data, using the per-trial
cells, saved per-step logs and layouts where possible.

Checked tables and macros:

- per-seed table: 9 rows;
- time-to-goal table: 6 rows; the Wilcoxon p-values match scipy's default `zero_method='wilcox'`;
- validation-vs-test table: 6 rows;
- stress cells: 9 rows;
- filter-off baseline: 54 cells;
- example selection: the two group-A rows are tied at the median, and the stated actor-index tie-break picks
  s1_late K = 2 trial 348;
- gate history, the MAPPO check, the engineering stress count and the H3 breakdown;
- the confirmatory anchor: all values equal `verdict.json`;
- citations: the 21 cite keys and the 21 bibliography items are the same set; the 11 Stage 2 entries and the
  two added titles match the prompt exactly; Breeden et al. and the 7 untouched entries are byte-identical to
  644fed5.

**One discrepancy found and fixed through the data path.** Baseline table, s0_late K = 3: COLLISION is exactly
57.5% (115/200), but it was printed as 57. The cause was float rounding (`0.575*100 = 57.4999...`).
`scripts/paper_numbers.py` now rounds that table with exact decimal arithmetic (half-even, the same as every
other tie in the table), and the cell reads 58. No other macro changed; the anchor check still passes.

No test suite was run: no code under `cbf_study/` changed in this run.
