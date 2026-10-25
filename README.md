# CBF-QP safety filter on a frozen formation policy: protocol and filter (VMAS 1.5.2)

Simulation-only harness for the EE 609 study: does an execution-time CBF-QP filter between a
frozen policy and `env.step()` reduce collisions, and at what cost in formation error, goal
completion and deadlock? **No trained policy is included.** `scenario.SlotPolicy` is an
obstacle-blind test double so the protocol and the filter can be exercised; no result from it
belongs in the study. All numbers (radii, limits, ranges, densities, goal and formation
tolerances) are placeholders pending advisor approval.

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
| `docs/CBF_DESIGN.md` | barrier, relative degree, footprint cover, safety lemma, assumptions, limits |
| `tests/` | 30 tests, including the constructed cases |

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

    pip install -r requirements.txt
    python -m pytest -q tests                     # about 4 minutes (VMAS and stress tests dominate)
    python scripts/run_paired.py --sim vmas --n-obst 5 --batch 100 --alpha 2.0 --out results/paired
    python scripts/run_stress.py --batch 100      # writes results/stress_summary.csv

## Things to decide or check before the study run

* The advisor-approved velocity bounds, radii, `ℓ`, ranges and tolerances replace the placeholders
  in `FilterConfig` and `ProtocolConfig`. They must be identical in both conditions (`check_consistent`).
* The frozen MAPPO actor replaces `SlotPolicy` (any callable `policy(poses, layout) -> [B,N,2]`).
  Its deterministic action goes in as `u_nom`; the filter never touches the policy.
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
