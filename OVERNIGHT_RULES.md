# Overnight run rules (copied verbatim from the instructions, 2026-10-26)

Start: Mon Oct 26 00:38:26 UTC 2026. Time cap 6 h; at 5 h 35 min elapsed skip to FINISH.
"Conservative" means: claim less and change less, log the choice in MORNING_SUMMARY.md, and keep going; it never means stopping the whole run.

HARD RULES
- No training. No simulation of study trials, no environment stepping on study layouts, no new layouts. Analyses read saved logs, saved layouts and CSVs only. Importing the filter's row-building functions and evaluating them at saved states is allowed and is not simulation.
- Never generate, read or use the reserved 5,000,000-5,999,999 block. Never regenerate any 4,000,000-block layout from seeds; use only saved files. If needed data is not saved, log "not computable from saved data" and disclose the missing coverage.
- Do not change any confirmatory number, the locked protocol, margins, filter code, scoring, tolerances, actors, config, or the layout sampler.
- The locked primary analysis keeps its confirmatory status. Only the clearance, mechanism, infeasibility and numerical diagnostics are labelled "exploratory, post hoc".
- Empirical results and computed constants must come from committed scripts reading committed data. Exempt: bibliographic years, dates, identifiers, algebraic constants, and unchanged historical reports.
- Before every commit: stage explicit paths only (never `git add -A` or `.`), then run `git diff --cached --name-status | grep '^D'`; if it prints anything, unstage and do not commit. Keep results/REPORT.md untracked. Commit and push (or bundle) after each phase.
- When a phase hits its time cap, commit what exists marked INCOMPLETE and move on.
- If a term is not defined in the glossary or the listed docs, log it and skip that item; do not infer.
- Do not install system packages and do not run anything that could trigger a permission prompt (no apt, no sudo). If a TeX engine is not already available, skip PDF compilation.

GLOSSARY
- Filter C: the existing CBF-QP filter (cbf_study/cbf_qp.py, solver cbf_study/qp2d.py) exactly as used in the final test, alpha* = 10.
- B-match: the brake comparator in cbf_study/bmatch.py (v = 0, clipped nominal omega, on the same barrier-row trigger at alpha*).
- Group A: final-test pairs where filter off reached GOAL and filter on ended in a filter-induced stall (n = 2,274; obstacle-labelled 2,259; neighbour-only 15). Group A': filter off GOAL and filter on not GOAL for any reason (n = 2,540). Group B: both reached GOAL (n = 11,255). As used in MECHANISM_EXPLORATORY.md section 4.
- A5: the design-document assumption that every barrier h >= 0 at the initial state.
- Stand-off: the barrier value h* = delta/alpha at which a zero command first satisfies a filter-C row (delta from the existing margin functions).

PHASES (caps): 1 clearance thresholds 30 min (+10 optional AUC); 2 infeasibility audit 45 min; 3 design-doc corrections 40 min; 4 paper draft 2 h 15 min; 5 verification 40 min; FINISH 20 min reserved.
Run `date` before every phase; re-read this file before every phase.
