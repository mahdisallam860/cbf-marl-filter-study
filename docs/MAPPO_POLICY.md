# Trained policy, gate and pilot: design

This replaces the stand-in `SlotPolicy` with a MAPPO policy trained **without the filter** and
frozen for evaluation. Every number referred to here is in `config/study.toml` and is a
placeholder pending advisor approval. Safety statements remain those of `docs/CBF_DESIGN.md`
(Lemma 1 under A1 to A9); nothing in the training, gate or pilot is evidence of continuous-time
safety.

## 1. Training environment (`cbf_study/train_env.py`)

The training environment matches evaluation by sharing code, not by re-implementing it:

| Element | Training | Evaluation |
|---|---|---|
| Physics | `TrainVmasEnv`, a subclass of `envs.VmasEnv` (DiffDrive, world drag 0, RK4, `collide=False`, same dt, radii, bounds) | `envs.VmasEnv` |
| Layouts | `scenario.sample_layout`, training seeds | `scenario.sample_layout`, validation / pilot seeds |
| Commands | raw action clipped to the velocity box | raw action clipped (filter off) or QP (filter on) |
| Collision | `protocol.score_motion` (exact arcs, swept rule, realised end poses), stop at first collision | the same function inside `protocol.run_trials` |
| Goal / timeout | `protocol.goal_reached`; `max_steps` | the same |

The only addition is `reset_subset`, which resets finished environments and leaves the VMAS state
of the others untouched (bit-identical, tested). A test runs the same deterministic policy on the
same layouts through the training environment and through `protocol.run_trials` and requires
identical outcomes and end steps for every trial.

Training mixes the three study densities (one VMAS batch per density, equal size). Seed ranges
for training, validation (gate, checkpoint selection) and pilot layouts are disjoint
(`config.train_seed`, `validation_seed`, `pilot_seed`; tested).

**Reward** per control period (weights in `[reward]`):

    w_progress (|c_k − g| − |c_{k+1} − g|) − w_formation · e_form(k+1) − w_control · mean_i[(v_i/v̄)² + (ω_i/ω̄)²]
      − w_collision_obst · [robot–obstacle collision] − w_collision_rr · [robot–robot collision]
      − w_proximity · Σ_pairs max(0, 1 − clearance / proximity_margin)            (config v2 only)

`c` is the team centroid, `g` the goal, `e_form` the protocol's formation error (mean slot error).
A collision ends the episode; a timeout is a truncation (bootstrapped with the critic).

**Observation** (`cbf_study/observations.py`), computed from `(poses, layout)` only, in each
robot's body frame: slot error relative to the centroid, centroid-to-goal vector (norm-clipped),
teammates' relative positions and headings, up to `k_obst` obstacles nearest first (relative
centre, surface gap, present flag; absent beyond `obs_range`), agent one-hot. The same function is
used by the training environment and by the frozen policy; it never sees anything the filter
computes. In config v1 `k_obst = 4`: at K = 5 and 8 every held-out layout had robots with more
obstacles inside `obs_range` than they observed. Config v2 sets `k_obst = 8` (the largest density),
and `tests/test_observation_sensing.py` checks on held-out layouts at every density that each robot
observes every obstacle within `obs_range`, in its body frame, nearest first, and nothing beyond.
`obs_range` (2.0 m, from the robot centre) is the policy's perception range; the filter's
`sensing_range` (1.0 m, from the barrier point) is a separate value.

## 2. MAPPO (`cbf_study/mappo.py`)

Compact PyTorch implementation (BenchMARL was considered: it would have needed a custom TorchRL
environment wrapping the protocol's own float64 scoring, stop rule and layout sampler anyway, so it
would only have contributed the PPO loop). Shared-parameter Gaussian actor whose mean is
`center + half · tanh(net(obs))`, so the deterministic action lies in the velocity box; state-
independent log-std; centralised critic on all robots' observations; GAE; clipped surrogate;
running value normalisation. Hyperparameters are in `[mappo]` and stored in every checkpoint.
Actor checkpoints every `checkpoint_every` iterations; curves in `results/train/seed_*/curve.csv`
(training rates there are for the *stochastic* policy).

**Config versions.** v1 (`config/archive/study_v1.toml`, env fingerprint `cf442ee5e99eb55a`) trained
`results/train/seed_*` and produced the failed gate in `results/gate/`; the v1 checkpoints load only
under v1. v2 (`config/study.toml`) changes the observation (`k_obst`), adds the proximity term, the
curriculum and the PPO recipe; the layout sampler, the filter and the outcome scoring are unchanged.

## 2a. Checks before retraining

* **Layout solvability** (`cbf_study/solvability.py`, `scripts/audit_solvability.py`,
  `results/solvability/`). Geometry only. FOUND = a constructive plan whose every segment is checked
  exactly against obstacles inflated by `R + R_o + 2 mm` and whose turn-in-place step count fits the
  400-step horizon; BLOCKED = start and goal disconnected in an over-approximation of the free
  space (a proof at any horizon); otherwise UNDETERMINED. Three questions: each robot alone, the
  team avoiding one another (2R + 2 mm), the team within the formation tolerance during the whole
  transit. The blocked proof for the third question is a weak relaxation (single-disc penetration
  depth), so that question rarely yields BLOCKED; it is reported as undetermined, not as blocked.
* **MAPPO on a stock task** (`scripts/validate_mappo_vmas.py`, `results/mappo_validation/`): the same
  `MappoLearner` and recipe on VMAS `navigation` (3 agents), evaluated on fixed episodes against the
  initial and a uniform-random policy; implementation checks in `tests/test_mappo_checks.py`
  (applied actions clipped to the box while log-probabilities use the raw sample consistently,
  ratio exactly 1 before the first gradient step, time-limit steps bootstrapped from the pre-reset
  state and terminal steps not, each actor sees only its own observation row).

## 2b. Curriculum (config v2, `scripts/train_curriculum.py`)

Stages K = 0 → 1 → 2 → 3 → (2, 5, 8) with env-step budgets fixed in `[curriculum]` before any run.
A stage ends when its budget is spent, or earlier when the held-out team GOAL rate (validation
layouts, frozen deterministic policy, filter off, the gate's evaluation path) reaches
`advance_goal_rate` at every stage density. **Stop rule:** if the held-out K = 2 GOAL rate is below
`k2_stop_goal_rate` when the K = 2 stage ends, training stops and diagnostics are reported. Each
density keeps one training-layout stream across stages. Only final-stage checkpoints (trained on
the study mixture) are written where the gate reads them.

## 3. Freezing (`cbf_study/policies.py`)

`FrozenMAPPOPolicy(ckpt)(poses, layout) -> [B,N,2]` returns the Gaussian mean clipped to the
velocity box; the actor is in eval mode with `requires_grad=False` and runs under
`torch.inference_mode`. Every evaluation asserts the parameter digest is unchanged afterwards. A
checkpoint is refused when its environment fingerprint differs from the current config.

## 4. Gate (pre-registered before training)

Filter OFF, `VmasEnv`, validation layouts, every checkpoint, every density.

* **STOP** if pooled GOAL rate < `gate.goal_rate_min` (the policy "rarely reaches the goal") or
  pooled COLLISION rate < `gate.collision_rate_min` ("near zero": no room for the filter to show a
  difference). Per-density rates are reported and flagged against the same thresholds.
* **Selection rule:** best checkpoint = highest pooled validation GOAL rate, ties to the later
  checkpoint. Collision rate is deliberately not a selection criterion.
* `run_pilot.py` refuses to run on a checkpoint that did not pass.

## 5. Pilot

Pilot layouts (disjoint from validation), paired filter-off/on at every density and every alpha in
`filter.alpha_sweep` (all satisfy `α·dt ≤ 1`). Filter-off is run once per density (it does not
depend on alpha, tested) and paired with each filter-on run. Outcomes GOAL / COLLISION /
INFEASIBLE / TIMEOUT, formation error, interventions, stalled (deadlock) flag. INFEASIBLE trials are
counted as failures exactly as they happen.

Statistics (`cbf_study/stats.py`): McNemar exact test with the Newcombe CI for COLLISION, GOAL and
stalled; Wilcoxon signed-rank with Hodges–Lehmann and bootstrap CIs for formation error (all pairs,
and pairs where both conditions reached the goal). Sample size: Connor (McNemar) and Noether
(Wilcoxon) closed forms, cross-checked by simulation from the pilot, at α = 0.05 and with a
Bonferroni correction over the density × alpha family.
