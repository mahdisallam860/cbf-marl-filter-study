# Provenance audit, 2026-10-25 (17:04–17:20 UTC)

Scope: record and back up the state, regenerate every recorded validation and pilot layout at the
commit each run used, verify curriculum seed 1's reconstructed records and resume state, run the
test suite. No training, no filter-on evaluation, no final test seeds, no change to config, sampler,
filter or scoring code. This file reports findings only.

Before the audit request arrived, two things had finished: curriculum seed 1 completed its planned
8M-env-step final stage (17,498,112 env steps in total), and one filter-off gate run over the 51
final-stage curriculum checkpoints (`results/gate_v2/`, validation layouts) completed.

## 1. Recorded state and backup

* HEAD `ff462684e6c68d5d2414d136cd7456e7a167c1f7`, pushed to `origin/claude/elegant-hopper-v8jxlv`
  (confirmed on GitHub). Annotated tag `audit-pre-2026-10-25` → that commit exists **locally only**:
  the session's git proxy refused the tag push (three attempts) and no tag-creation API is available
  here. The tag is inside the backup's git bundle.
* Uncommitted at tag time, by instruction still uncommitted: `results/curriculum/`, `results/gate_v2/`,
  the `README.md` edit, the `results/REPORT.md` draft (not continued).
* Backup `audit_backup_20261025.tar.gz`, 48,820,039 bytes, SHA-256
  `f06dc0070432e6b5452099ab2c6c6d7cb6157855e966487b00778b78e48e54bc`; 393 files with a `SHA256SUMS`
  manifest: the whole `results/` tree (incl. untracked `latest.pt` states), `config/`, `docs/`,
  `README.md`, `repo.bundle` (all refs incl. the tag; `git bundle verify` OK), raw session logs (stdout
  of every curriculum process, the seed-1 resume, the gate-v2 run, the rebuild), seed 1's pre-patch
  `latest.pt`, test logs, `ENVIRONMENT.txt` (pip freeze), `STATE.txt`.
* Verification: archive hash OK; extracted into a fresh directory, all 393 manifest entries verified;
  snapshot byte-identical to the live `results/`, `config/`, `docs/`, `README.md`. Snapshot and
  archive made read-only in the container. Delivered to the user as a downloadable file (outside the
  session); the delivered copy can only be verified on the receiving side
  (`sha256sum -c audit_backup_20261025.tar.gz.sha256`, then `sha256sum -c SHA256SUMS`).

## 2. Layout regeneration (`regenerate_layouts.py`, `_regen_child.py`)

Run → commit whose code it ran (first output written seconds after the commit; the sampler files of
the working tree at launch equal that commit's):

| run | commit | scenario.py | study.toml | config / env fingerprint |
|---|---|---|---|---|
| v1 training phase 1 (first output 14:03:18) | 871206b | 0c8bf76 (v1) | 922aeeb | v1 / cf442ee5e99eb55a |
| v1 training phase 2 (14:12:46) | a99fcc2 | 0c8bf76 | 922aeeb | v1 |
| v1 gate (14:38:40) | 1280a8a | 0c8bf76 | 922aeeb | v1 |
| solvability audit (14:58:23) | ba87dbf | 0c8bf76 | 922aeeb | v1 |
| curriculum, all seeds (15:15:15) | 5b2820f | 0c8bf76 | 520f8c1 | v2 / c60352aee3f11ed8 |
| curriculum seed-1 resume (16:33:23) | d112bc3 | c7a6061 (v1.1) | 520f8c1 | v2 |
| v2 gate (16:57:31) | ff46268 (HEAD) | c7a6061 | 4a4a01e (comment only) | v2 |

Method: read-only detached worktrees at all seven commits; the regeneration ran as the unprivileged
user `nobody` (so the read-only bits bind) with `python -I` and only the worktree providing
`cbf_study` (asserted), each commit's own config, in this repository's pinned venv (numpy 2.5.4,
torch 2.14.1, vmas 1.5.2). Sets: validation (n = 200) and pilot (n = 500), K ∈ {0, 1, 2, 3, 5, 8}
(K = 0 added: the curriculum's stage-0 held-out evaluations used it).

Results (`layout_audit_summary.json`):
* 84 set comparisons (7 commits × 12 sets) against HEAD: **0 differences** (dtype, shape and bytes of
  `init_poses`, `obst_xy`, `goal`, `slots`). Per-array SHA-256 in `layout_hashes_by_commit.csv`.
* 45 recorded layout IDs (by run, run seed, split, K; `recorded_layout_ids.csv`): layout seed and set
  size match the formula in every case, and the regenerated set at the run's launch commit is
  byte-identical to HEAD's. Note: the `eval_layout_seed` column of the v1 gate table was added after
  that run, from the formula (validation_base + K), not written by the run.
* Hashes recorded before the sampler change (`recorded_hashes_pre_v1.1.json`, written 16:30:09 UTC by
  regeneration under sampler v1, not by the runs): 42 checks, 0 mismatches.
* Outcome cross-check (`layout_outcome_crosscheck.csv`): for all 51 final-stage checkpoints × 3
  densities, the curriculum's held-out counts equal the v2 gate's counts (153 rows, 0 mismatches). For
  seeds 0 and 2 the held-out evaluations ran at run time under sampler v1 and the gate under v1.1.
* `layouts_HEAD/*.npz`: the regenerated arrays at HEAD, saved for future comparisons.

**Limit:** no run saved the layouts it used. Regeneration shows that the code at each launch commit
produces these arrays now under the pinned environment; it cannot prove that the historical runs
used identical arrays. The outcome cross-checks are consistent with identity but are not a proof.
Not regenerated: training-layout streams; `route_geometry.csv` (records no layout seed; it used the
validation sets, n = 200); results from the original commit 728aaf2 (sampler before
`ScenarioConfig` noise fields; no validation or pilot sets existed then).

## 3. Curriculum seed 1 (`verify_seed1.py`, `seed1_verification.json`)

Seed 1 crashed at global iteration 565 (sampler stall on training chunk seed 1108377) and was resumed
from the state its own run saved at iteration 546.

* Curve: 712 rows, 546 reconstructed from `train.log` (iterations 1–546), 166 written by the resumed
  run. Reconstructed rows vs `train.log`: 0 mismatches; vs the original process's independent stdout
  copy: 0 mismatches. All 712 rows satisfy env_steps = iteration × 24,576 and are contiguous.
  Iterations 547–565 of the original run (466,944 env steps) were lost in the crash and are not
  counted in the totals.
* Stage checkpoints: 38, metadata consistent with the curve; the 29 written before the crash are
  exactly those with iteration ≤ 546 (`seed1_checkpoints.csv`).
* Held-out: 72 rows, 45 reconstructed by re-evaluating saved checkpoints, 27 written by the resumed
  run. Re-evaluated again now in fresh processes: identical in every field (0 mismatches); vs the
  goal/collision/timeout values the original run logged: 0 mismatches.
* Resume state: the original run's saved state (pre-patch copy) has an actor identical to its
  iteration-546 checkpoint. Loading it reproduces, bit for bit, the actor, critic, both Adam states
  (step 87,360 = 546 × 160), the value normaliser and the action-sampling generator state; the logged
  resume position (stage 4, 3,932,160 stage env steps, 13,418,496 total, iteration 546) equals the
  saved one. The file the resume actually loaded (saved state + reconstructed stage summaries) was
  later overwritten by the resumed run; adding the summaries was shown to leave every state entry
  unchanged. Not restored on resume: the environments and their in-progress episodes, the position
  inside the partly consumed training-layout chunk (resumed at the next chunk), the global torch/numpy
  RNG (used for neither action sampling nor layouts), and iterations 547–565.
* Labels: `seed1_provenance.csv` (789 rows: curve, held-out, stage summaries → `original_run_output` or
  `reconstructed_*`, with the check applied). In the working tree, seed 1's `curve.csv` (`source`),
  `heldout.csv` (`provenance`) and `stages.json` (`provenance`) carry the same labels; their
  post-label hashes are in `seed1_labelled_files_sha256.json`, the pre-label versions in the backup.

## 4. Test suite

`python -m pytest -q tests` at HEAD ff46268 (no uncommitted code, test, script or config changes):
**80 passed, 0 failed** in 210 s (`test_suite_run.txt`).

## Left unchecked

* The delivered backup copy (verifiable only on the receiving side) and the remote tag (push refused).
* Historical identity of the layouts (not provable without saved layouts; see §2).
* Training-layout streams of every run (only validation and pilot sets were in scope).
