# Independent recomputation of the paper's numbers (overnight phase 5)

Performed by a fresh subagent given only `paper/main.tex`, `paper/numbers.tex`, committed result files
and `docs/methodology/`; it did not open `scripts/*.py`, ran no simulation, and changed nothing. Its own
scripts recomputed the numbers from the per-trial records (`results/final_test/cells/*.csv.gz`, 81,000
rows = 9 actors x 3 densities x 1,000 trials x 3 conditions; pairs matched on actor, density, trial).

Result: **no numeric discrepancy** in the abstract, the confirmatory table (failure rates, (a) difference,
Newcombe bounds and 95% intervals, exact one-sided p-values, GOAL rates and loss with bounds, formation
increase with bootstrap and normal bounds, eligible/excluded counts, verdicts) or the discussion (alpha*,
B-match GOAL gaps 1.0-3.2 pp, group A median clearance 3.3 cm, 433 infeasible trials, alpha = 0.5
selection counts). The verifier's own bootstrap (B = 4,000) gave 0.0290 for the K = 3 bound against the
locked 0.0291 (resampling noise; normal cross-check 0.0291). Its independent Newcombe method-10 code agreed
to about 6e-6 on the proportion scale.

Citations: all 12 cite keys have a bibliography item and all items are cited; all 12 URLs appear verbatim
in `docs/methodology/` (ORCA only in the ZR audit); every venue/year stated is in those documents.

Wording points raised and applied through the text (no number changed): "removed almost all collisions"
replaced by the zero filter-on collision counts (macros); "infeasible quadratic programmes" replaced by
"infeasible trials ... binding barrier row". Not applied: the plain-text "alpha*dt = 1" sentence, which
the instructions require verbatim.
