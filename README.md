# CBF-QP safety filter on a frozen formation policy: protocol, filter and MAPPO policy (VMAS 1.5.2)

Simulation-only harness for the EE 609 study: does an execution-time CBF-QP filter between a
frozen policy and `env.step()` reduce collisions, and at what cost in formation error, goal
completion and deadlock? The policy is a MAPPO actor trained **without** the filter and frozen
for evaluation (`cbf_study/policies.py`); `scenario.SlotPolicy` remains only as an obstacle-blind
test double for the protocol tests. Every number (radii, limits, `ℓ`, ranges, densities, goal and
formation tolerances, reward and collision-penalty weights, gate thresholds) lives in
`config/study.toml` and is a placeholder pending advisor approval.

**Current status: STOP at the filter-off gate** (`results/REPORT.md`). No trained policy (config v1,
or the config v2 curriculum) reaches the goal at K = 5 or K = 8, although every held-out layout is
solvable; no filter-on result has been produced.

## Layout

| Path | Contents |
|---|---|
| `cbf_study/geometry.py` | exact unicycle arcs, swept-path (between-step) separation bound, exact-arc test oracle |
| `cbf_study/qp2d.py` | exact batched 2-variable QP solver (no iteration limit) |
| `cbf_study/cbf_qp.py` | barrier, margin, `FilterConfig`, `CBFQPFilter` |
| `cbf_study/protocol.py` | shared run loop, outcomes, scoring, CSV logging, `run_paired` |
| `cbf_study/envs.py` | `VmasEnv` (pinned VMAS 1.5.2, drag 0, RK4, `collide=False`) and `KinematicEnv` |
| `cbf_study/scenario.py` | seeded layouts, stand-in policy |
| `cbf_study/stress.py`, `scripts/run_stress.py` | adversarial sweep, writes `results/stress_summary.csv` |
| `scripts/run_paired.py` | paired off/on run with CSV logs |
| `scripts/gate_test_vmas_diffdrive.py` | the earlier simulator-selection gate test |
| `config/study.toml`, `cbf_study/config.py` | the single study config, v2 (every value marked pending approval), `check_paired`, env fingerprint; v1 in `config/archive/` |
| `cbf_study/observations.py` | policy observation from `(poses, layout)`, shared by training and evaluation |
| `cbf_study/train_env.py` | VMAS training env: `VmasEnv` subclass, `sample_layout`, `protocol.score_motion`, stop at first collision |
| `cbf_study/mappo.py`, `scripts/train_mappo.py` | compact PyTorch MAPPO, checkpoints and curves under `results/train/` |
| `cbf_study/policies.py` | `FrozenMAPPOPolicy(ckpt)(poses, layout) -> [B,N,2]`, eval mode, no gradients |
| `cbf_study/evaluate.py`, `scripts/gate_eval.py` | filter-off gate per checkpoint, checkpoint selection (`results/gate/`) |
| `cbf_study/stats.py`, `scripts/run_pilot.py` | paired tests, CIs, sample size; pilot runs (refuses without a passed gate) |
| `cbf_study/solvability.py`, `scripts/audit_solvability.py`, `scripts/route_geometry.py` | layout solvability audit (`results/solvability/`) |
| `cbf_study/stock_env.py`, `scripts/validate_mappo_vmas.py` | MAPPO check on stock VMAS `navigation` (`results/mappo_validation/`) |
| `scripts/train_curriculum.py`, `scripts/rebuild_curriculum_logs.py` | curriculum K 0→1→2→3→(2,5,8) with pre-set budgets and the K=2 stop rule (`results/curriculum/`) |
| `cbf_study/bmatch.py` | B-match brake comparator (locked protocol D5): same barrier rows as the filter at a given gain, v = 0 when violated, no QP; protocol adapter with separate actor / applied / brake logs |
| `cbf_study/trial_metrics.py` | pass-through trajectory recorder, executed-period counts, common-prefix formation difference, reached-goal-region, sampled barrier minima |
| `cbf_study/locked_stats.py` | confirmatory statistics of the locked protocol (one-sided McNemar, Newcombe bounds, equal-policy stratified bootstrap) |
| `docs/PREFILTER_VALIDATION_PROTOCOL_LOCKED_2026-10-25.md`, `config/evaluation_protocol_locked.toml` | author-locked (not advisor-approved) evaluation protocol |
| `docs/CBF_DESIGN.md` | barrier, relative degree, footprint cover, safety lemma, assumptions, limits |
| `docs/MAPPO_POLICY.md` | training env, reward, observation, MAPPO, freezing, gate and pilot design |
| `tests/` | protocol, filter, config, training env, MAPPO, frozen policy, statistics |

## Protocol fixes in this version

1. **Between-step scoring.** Motion in each control period is scored on the exact arc, with a
   conservative swept rule (64 sub-samples plus a Lipschitz correction, plus the simulator's
   realised end poses). A constructed case where both sampled poses straddle an obstacle without
   overlap is caught (`tests/test_protocol.py::test_between_step_crossing_*`,
   `tests/test_geometry.py`).
2. **Stop at the first geometric collision.** Distinct outcomes `GOAL`, `COLLISION`,
   `INFEASIBLE`, `TIMEOUT`; success is `GOAL` only; the other three are task failures; same
   rules and same code path for filter-on and filter-off.

## Run

Python >= 3.11 (`tomllib`). On Debian/Ubuntu the system setuptools cannot build the vmas 1.5.2
sdist (`install_layout` error); use a virtual environment with a current setuptools:

    python -m venv venv && . venv/bin/activate && pip install -U pip setuptools wheel
    pip install -r requirements.txt
    python -m pytest -q tests                     # about 4 minutes (VMAS and stress tests dominate)

Study pipeline (all values from `config/study.toml`):

    python scripts/audit_solvability.py           # geometry-only solvability of the held-out layouts
    python scripts/validate_mappo_vmas.py         # MAPPO on stock VMAS navigation
    python scripts/train_curriculum.py --seed 0   # config v2 curriculum, filter OFF; results/curriculum/seed_0/
    python scripts/train_mappo.py --seed 0        # no curriculum (the v1 runs in results/train/ used config v1)
    python scripts/gate_eval.py --ckpts "results/curriculum/seed_*/checkpoints/ckpt_it*.pt" --out results/gate_v2
    python scripts/gate_eval.py --config config/archive/study_v1.toml   # reproduce the v1 gate (results/gate/)
    python scripts/run_pilot.py --gate results/gate_v2/selection.json --ckpt <selected checkpoint> --tag <name>   # refuses unless the gate passed
    python scripts/run_paired.py --ckpt <checkpoint> --n-obst 5 --alpha 2.0 --out results/paired
    python scripts/run_stress.py --batch 100      # adversarial sweep, results/stress_summary.csv

## Things to decide or check before the study run

* The advisor-approved values replace the placeholders in `config/study.toml` (set `approval` to
  `"approved"`). Filter-on and filter-off are built from that one file and may differ in `alpha`
  only (`config.check_paired`). Changing an environment value invalidates trained checkpoints
  (env fingerprint), so approve those before the final training run.
* The frozen MAPPO actor (`policies.FrozenMAPPOPolicy`) is the nominal policy; its deterministic
  action (the mean, clipped to the velocity box) goes in as `u_nom`; the filter never touches the policy.
* The safety margin makes the filter conservative (docs section 4): robot–robot stand-off is larger
  than obstacle stand-off, and small `α` is much more conservative. Choose the barrier-gain range for
  the sweep from a pilot, with `α·dt ≤ 1`.
* Infeasible steps are a measured outcome. In the adversarial sweep they were concentrated at the
  smallest gain (231 of 600 trials at `α = 0.5`, 1 of 2 400 at `α = 2`, none at `α = 5` or `10`);
  for the real policy the rate is an empirical question.
* The baseline policy must collide often enough in the chosen densities for the comparison to have
  room to show a difference (ceiling effect noted in Stage 2).
* Safety statements are limited to docs section 4 under assumptions A1 to A9. The numerical tests
  are evidence that the code matches that statement, not proof of continuous-time safety.
