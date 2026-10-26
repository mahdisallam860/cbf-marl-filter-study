# Morning summary: overnight run of 2026-10-26

- **Start:** 00:38:26 UTC, as reported by `date`.
- **End:** about 00:54 UTC, when this file was written.
- **Elapsed:** the container clock shows about 16 minutes. All phases finished well inside the 6 h cap; no time-cap skip occurred.
- **Push status:** every commit was pushed to `origin/claude/elegant-hopper-v8jxlv` (no PUSH_BLOCKED, no bundle).
- `results/REPORT.md` remains untracked.

## Step 0 (setup)

- `git pull --ff-only` succeeded (HEAD was 02f0434).
- **Baseline gate: PASSED.** `git diff --name-status c2e331a HEAD` shows exactly the three `docs/methodology/` files.
  - The history in between contains `77f339c`, which deleted the mechanism-analysis files, and `02f0434`, which restored them.
  - The net diff is clean.
- **Push dry-run:** OK.
- **TeX:** no TeX engine was available, so no PDF was compiled.
- **Rules file:** OVERNIGHT_RULES.md was committed in `c09ca30`.

## Phases

| phase | status | commit | outputs |
|---|---|---|---|
| 1 Clearance thresholds (exploratory, post hoc) | done, incl. optional AUC | `b0a8aaa` | MECHANISM_EXPLORATORY.md section 5; `results/final_test/mechanism/clearance_thresholds*.csv`; `scripts/clearance_thresholds.py` |
| 2 Infeasibility audit (exploratory, post hoc) | done | `e151de1` | `results/final_test/INFEASIBILITY_AUDIT.md`; `results/final_test/infeasibility/`; `scripts/infeasibility_audit*.py` |
| 3 Design-document corrections | done (docs only) | `60f3714` | `docs/CBF_DESIGN.md` section 4a, A5 exception, B-match note; README status; `scripts/design_doc_constants.py`, `results/design/margin_constants.json` |
| 4 Paper draft | done; PDF not compiled (no TeX) | `01942dc` (plus wording fixes in `b0296d2`) | `paper/main.tex`, `paper/numbers.tex`, `paper/figures/F1-F5.pdf`, `scripts/paper_numbers.py`, `scripts/paper_figures.py` |
| 5 Verification | done | `b0296d2` | `results/overnight/independent_verification.md`, `results/overnight/test_suite_run.txt` |

## Key results

**Phase 1 (pooled K = 1–3, sampled minima).**
- 51.1% of all trials reached GOAL unfiltered.
- 43.1% did so while keeping at least 5 cm of obstacle clearance.
- The filter-on GOAL rate on the same layouts was 42.7%.
- AUC of the filter-off minimum obstacle clearance for "filter on not GOAL" is 0.96 (descriptive).
- This describes how close the unfiltered runs came; it does not measure what a buffer or a filter would cost.

**Phase 2a reproduction count: 433 of 433 INFEASIBLE trials reproduced** by the unchanged filter code at the saved states.
- In all of them the binding row sits at its stand-off h* = delta/alpha, with |h − h*| below 4.3e-8 m².
- The zero-command slack is −4.9e-6 to −8.4e-9.
- 410 of 433 meet the pre-declared near-boundary diagnostic. This is consistent with numerical sensitivity near the boundary; it is not proof of the cause.
- In 88 of 433, float32 arithmetic flips the sign of the binding zero-command slack.

**Phase 2b (A5).**
- Exactly 1 of 27,000 final-test trials (s2_early, K = 3, trial 646) starts with h_o = −0.00777 m², while its body clearance is 0.087 m.
- 0 of the 1,200 saved validation layouts start with a negative barrier.
- Validation K = 4 is not computable from saved data (its layouts were not saved).
- Likely cause: the sampler checks obstacles against the nominal start centroid, not against the noisy start poses.
- It is recorded as an A5 exception. The sampler is unchanged.

**Anchor check: PASSED.** The confirmatory macros are identical whether built from `verdict.json` or from `endpoints.csv`.

**Independent verification:** a fresh subagent, which did not use `scripts/paper_numbers.py`, recomputed every number in the abstract, the confirmatory table and the discussion.
- It worked from the per-trial records and found no numeric discrepancy.
- All 12 citations trace to URLs in `docs/methodology/`.
- Two wording issues were fixed in the text; no number changed:
  - "removed almost all collisions" was replaced by the zero filter-on collision counts;
  - "infeasible quadratic programmes" became "infeasible trials".

**Tests** (synthetic fixtures; the two stress-harness tests were skipped by rule): 113 passed, 1 failed.
- The failure is `tests/test_seed_reservation.py::test_no_saved_result_or_audit_record_uses_a_reserved_seed`.
- It predates the authorized final test, and it now flags the final test's legitimate 4,000,000-block seeds in `results/final_test/`.
- No 5,000,000-block seed appears anywhere.
- The test was left unchanged.

## Judgement calls made overnight

1. **Phase 2a QP call.** "Reproduced as infeasible" was determined by calling the unchanged `CBFQPFilter`, which runs its QP solver, once per saved state. Only the verdict was compared with the logged one; no outcome was relabelled and no solution was used. I read "do not re-solve" as "do not relabel or substitute".
2. **Phase 2a dtypes.** The filter code promotes the logged float32 values to float64, so the "logged dtype" and "float64" evaluations coincide on that code path. I added an independent float32-arithmetic evaluation of the same row formulas.
3. **Stand-off for pair rows.** h* = (δ_ij/2)/(α/2) = δ_ij/α, i.e. the row's margin divided by the row's gain.
4. **Breeden et al. Theorem 3** could not be checked, because the arXiv text was blocked (proxy 403). The design document and the paper say "citation to be verified" / "theorem numbering to be verified". The same applies to the theorem and section numbers of the ZR-remark sources.
5. **matplotlib** was pip-installed into the project venv. It is not a system package and needed no sudo; PyPI is reachable through the proxy. `requirements.txt` was not changed.
6. **Paper bibliography.** It is an inline `thebibliography` that uses only author/venue/year/URL as stated in `docs/methodology/`. No titles or pages were added, because the documents do not give them.
7. **Verbatim boundary sentence.** The sentence "At alpha*dt = 1, ..." was kept verbatim in plain text in the design document and the paper, as instructed. The verifier had suggested math formatting.
8. **Failing seed-reservation test.** It was left unmodified, as the conservative choice.
9. **Phase 4 counts and percentages** (for example the Phase 1 rates and the stall-class shares) are all macros generated from committed CSVs. The paper uses no hand-typed results.

## Skipped and why

- **PDF compilation:** no TeX engine is available.
- **Swept-arc clearance estimates:** optional, not done.
- **ZR-0 screen:** deferred earlier and not attempted overnight (not part of these phases).
- **Validation K = 4 initial-state check:** not computable from saved data.

## Decisions that need me / my advisors

1. **Paper.** Framing, title, authorship and the AI-assistance statement need settling, and so does whether to submit anything at all. The draft is marked "DRAFT - not for distribution".
2. **Margins and protocol.** Advisor review of the author-chosen margins and of the protocol is still pending; no advisor agreement exists.
3. **Breeden citation.** Verify the Theorem 3 citation and the theorem/section numbers of the ZR-remark sources against the papers.
4. **Seed-reservation test.** Decide whether to update it so it exempts the final-test outputs while still guarding the 5,000,000 block.
5. **Sampler.** Decide whether to fix the sampler's noisy-start obstacle check for future layout blocks. This would affect only new layouts and must never touch the used ones.
6. **Tolerance rule.** Decide whether a numerical tolerance rule at the stand-off belongs in any future method study. That study would use the untouched 5,000,000 block under a separately locked protocol, and must disclose that these post-test findings informed it.
7. **`results/REPORT.md`.** Decide what to do with the old draft, which is still untracked.
