# Verification of the on-vs-B-match and lost/gained outputs

**Result: PASS**, 182,537 checks, 0 failed.

`scripts/verify_bmatch_transitions.py` does not import `cbf_study`, `scripts/paper_bmatch_transitions.py`
or any project statistics helper. It recomputes the outputs from the saved final-test data:

- **Wilson limits and Newcombe methods 10 and 8:** its own implementation, with the normal quantile from
  the Python standard library. The interval endpoints agree to 1e-12, and the 2×2 counts agree exactly.
- **Outcomes:** taken from the NPZ `outcome` arrays in `results/final_test/steps/`, not from the cell CSVs.
- **Formation error e(t):** recomputed from the NPZ poses, with slots rebuilt from the protocol geometry
  (radius 0.30 m, angles π/2 + 2πr/3). It uses the common prefix L = min(T_on, T_bmatch), and estimates
  agree to 1e-8 m.
- **Bootstrap:** its own implementation of the locked draw order (B = 20,000, seed 20261026). Agreement:
  **bit-identical** (max |difference| 0 m).
- **LOST, GAINED and NET:**
  - recounted per density, seed, actor and K = 1–3 pooled;
  - LOST − GAINED equals the off GOAL count minus the on GOAL count in every scope;
  - (LOST − GAINED)/n equals the unrounded criterion-(b) estimate in `verdict.json` to 1e-12.
- **4×4 transitions:** all 48 cells recounted.
- **TIMEOUT classes:** `timeout_classes.csv` covers every filter-on TIMEOUT exactly once.
- **Data:** no duplicate keys; 9 actors × 1,000 trials × 3 arms per density; layout hashes identical
  across arms; seeds agree with `checkpoint_selection.json`.
- **Constructed fixtures:** all-negative, all-positive, zero-discordance, one-direction, negative association
  and arm reversal.

**Limitation:** the reached-goal-region split uses the saved filter-on flags `reached_goal_region`. Only
their aggregation is verified here, not the flags themselves.

Pooled K = 1–3:

- LOST 2540, GAINED 274,
  net 2266.
- LOST pairs whose filtered run never reached the goal region:
  2300.
