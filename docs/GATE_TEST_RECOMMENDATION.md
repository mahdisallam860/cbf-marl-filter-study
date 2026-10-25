# VMAS DiffDrive gate test: short recommendation

Scope: single-step dynamics fidelity and geometric collision scoring only. No MAPPO, no CBF, no safety claim.
Pinned: vmas 1.5.2, torch 2.14.1, numpy 2.5.4, CPU, seed 20261025. Run: `python gate_test_vmas_diffdrive.py --out results`.
Placeholders (not from your study): v_max 0.5 m/s, w_max 1.0 rad/s, robot radius 0.10 m, obstacle radius 0.20 m.

## What was found

1. **With VMAS default world drag (0.25), DiffDrive does not execute the commanded unicycle step.**
   Reading `diff_drive.py` and `World._integrate_state` shows why: VMAS multiplies velocity by (1 - drag) before adding the
   DiffDrive force, so realized velocity = commanded - drag x previous velocity.
   - Step 0 from rest: exact (position error < 6e-8 m).
   - Later steps (dt 0.1, RK4): position error up to 1.24e-2 m (const cmds) and 1.28e-2 m (random cmds), median ~5e-3 m,
     about 0.25 x v_max x dt. Heading error up to 0.025 and 0.033 rad.
   - Steady state realized speed is about commanded / (1 + drag), i.e. 80% of the command.
   - Over 40 steps the robot ends up to 0.39 m behind the kinematic reference, and the first geometric collision
     happens 3 to 5 steps later than the kinematic prediction.
2. **With world drag = 0 and RK4 integration, DiffDrive matches the closed-form unicycle to float32 precision.**
   Max position error 8.3e-8 m and max heading error 2.4e-7 rad over 6,000 steps per config, at dt 0.1 and 0.05.
3. **Euler integration** differs from the exact arc by up to 2.5e-3 m (dt 0.1) and 6.2e-4 m (dt 0.05) per step. Use RK4.
4. **Geometric collision scoring is independent of VMAS contacts and agrees.** With drag 0, first-event step and
   collision outcome were identical for the kinematic reference, VMAS with contacts off, and VMAS with contacts on in all
   62 cases (0 outcome mismatches). Boundary cases behave as expected: separation -0.0067 m scored a collision,
   +0.0067 m did not.
5. **VMAS contact response is soft and begins after the geometric event.** Robots overlapped by up to 0.05 m
   (robot-obstacle, sum of radii 0.30 m) and up to 0.10 to 0.12 m (robot-robot, sum 0.20 m). Contact response first moved a
   robot 1 to 2 steps after the first geometric event, had no measurable effect (< 1 mm) at separations of +0.0067 m or more,
   and 0.7 mm at exactly zero separation. After contact, a robot that keeps commanding forward is held at the boundary while the
   kinematic reference passes through, giving departures up to 1.25 m by step 40 in head-on cases.

## Recommendation

- **Do not use VMAS with default settings.** Default drag silently changes executed velocity by up to 20%, which would
  confound "filter failure" with "simulator did not execute the command".
- **VMAS is acceptable only in this configuration:** `World(drag=0)`, `DiffDrive(integration="rk4")`, agents and
  obstacles with `collide=False`, your own geometric collision scoring and termination, and your own command clipping.
  In that configuration VMAS is effectively an exact batched kinematic integrator.
- **If you do not need BenchMARL/TorchRL integration, a ~60-line own unicycle step is equivalent** and removes the dependency.
  If you plan to train MAPPO through BenchMARL, use VMAS in the configuration above.
- Keep the commanded-versus-realized check as a regression test whenever VMAS or the scenario changes.

## Limits of this test

- Straight-line approaches for Part B, so drag changed timing but not path. Curved or turning approaches were tested only for dynamics (Part A).
- Collision events are sampled at control steps (relative speed 1.0 m/s x dt 0.1 s = 0.1 m per step, small against the 0.2 to 0.3 m sums of radii). Faster or smaller robots need an inter-step (segment) check.
- Placeholder sizes and speeds; one CPU; one VMAS version. `u_range` is asserted by VMAS, not clipped, unless `clamp_actions=True`.
- Nothing here tests a CBF, a discrete-time safety guarantee, sensing, or learning.
