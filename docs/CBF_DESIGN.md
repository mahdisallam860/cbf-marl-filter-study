# CBF-QP safety filter: design, assumptions, and what is and is not claimed

Scope: three or more differential-drive robots, static circular obstacles, simulation only
(pinned VMAS 1.5.2, drag 0, RK4, `collide=False`). All numbers in code are placeholders until the
advisors approve the limits, formation and densities.

**Claim boundary.** Section 4 proves a *conditional, sampled-data* safety statement for the
barrier sets defined below, under assumptions A1 to A9. The numerical tests check that the code
implements that statement and that nothing in a large adversarial sweep contradicts it. They do
not establish continuous-time safety, and nothing here says the filter keeps the real robots
collision-free outside the stated assumptions.

---

## 1. Model and barrier point

Axle-centre pose `x = (x, y, θ)`, body = disc of radius `R` about `x`, control `u = (v, ω)`:

    ẋ = v cos θ,   ẏ = v sin θ,   θ̇ = ω

Barrier point, offset `ℓ > 0` ahead of the axle:

    p = (x + ℓ cos θ,  y + ℓ sin θ)
    ṗ = J(θ) u,    J(θ) = [[cos θ, −ℓ sin θ], [sin θ, ℓ cos θ]] = Rot(θ) · diag(1, ℓ)

`det J = ℓ`, so `J` is invertible for every `θ` when `ℓ > 0` (checked in
`test_pdot_equals_J_u_and_relative_degree_one`).

## 2. Barriers and relative degree

    robot–obstacle:  h_o(x, c)   = |p − c|²        − D_o² ,   D_o  = R + ℓ + R_o
    robot–robot:     h_ij(x_i,x_j) = |p_i − p_j|²  − D_rr²,   D_rr = 2 (R + ℓ)

Relative degree. Along the dynamics

    ḣ_o = 2 (p − c)ᵀ J(θ) u = g_o(x)·u,      g_o = 2 Jᵀ (p − c)

`u` appears in the first derivative with a coefficient row `g_o` that is non-zero whenever
`p ≠ c` (because `J` is invertible and `|Jᵀq| ≥ min(1, ℓ)|q|`). So the **relative degree is 1
with respect to `(v, ω)`** for both barrier types, and a first-order barrier condition
`ḣ + αh ≥ …` can be imposed directly on the commands. This is why the barrier is on the offset
point: with `ℓ = 0` the matrix `J` has rank 1 (`ω` drops out of `ṗ`), `g` loses a control
direction, and the same construction does not give relative degree 1 for `(v, ω)`. Configurations
with `ℓ ≤ 0` are rejected by `FilterConfig.validate()`.

Robot–robot: `ḣ_ij = g_i·u_i + g_j·u_j` with `g_i = 2 J_iᵀ (p_i − p_j)` and
`g_j = 2 J_jᵀ (p_j − p_i)`, so each robot's control enters with relative degree 1.

## 3. How the geometry protects the full circular footprint

Because `|x − p| = ℓ`, the body disc is inside a larger disc centred on the barrier point:

    disc(x, R)  ⊆  disc(p, R + ℓ)

* **Robot–obstacle.** If `|p − c| ≥ R + ℓ + R_o` then every body point is within `R + ℓ` of `p`,
  hence at least `R_o` from `c`: the body clears the obstacle for every heading.
* **Robot–robot.** If `|p_i − p_j| ≥ 2(R + ℓ)` then, since `|x_i − p_i| = |x_j − p_j| = ℓ`,
  `|x_i − x_j| ≥ |p_i − p_j| − 2ℓ ≥ 2R`: the two body discs are disjoint.

So `h ≥ 0` is sufficient for the full footprints to be separated, regardless of orientation
(random check over 20 000 poses in `test_footprint_cover_robot_obstacle_and_robot_robot`).
The price is conservatism: the cover disc is larger than the body by `ℓ`, and the protected
region is not rotationally tight (at the back of the robot the body already touches the cover
disc, in front it leaves a gap of `ℓ`). Smaller `ℓ` gives a tighter cover, but `ω` then moves
`p` less (`|ṗ|` from rotation is `ℓ|ω|`), so avoiding an obstacle by turning needs more `ω`.

## 4. One-period (zero-order-hold) safety lemma

The policy runs at period `Δ` and the command is held constant over each period (ZOH). Imposing
`ḣ + αh ≥ 0` only at the sample instant does not keep `h ≥ 0` between samples, because `J(θ(t))`
and `p(t)` change while `u` is frozen. The constraint is therefore tightened by a margin.

**Bound on the second derivative.** With `u` constant over a period,
`ṗ = J(θ(t))u`, `|ṗ|² = v² + ℓ²ω² ≤ V_p²` and `p̈ = ω ∂J/∂θ u`, `|p̈| = |ω||ṗ| ≤ A_p`, where

    V_p = sqrt(v_abs_max² + ℓ² ω_max²),    A_p = ω_max · V_p .

For `h_o = |p − c|² − D_o²`: `ḧ_o = 2|ṗ|² + 2(p − c)·p̈`, so, with `d_k = |p(t_k) − c|` and
`|p(t) − c| ≤ d_k + V_p Δ` inside the period,

    |ḧ_o| ≤ M_o(d_k) = 2 V_p² + 2 (d_k + V_p Δ) A_p .

For `h_ij`: relative speed `≤ 2V_p`, relative acceleration `≤ 2A_p`, hence

    |ḧ_ij| ≤ M_r(d_k) = 8 V_p² + 4 (d_k + 2 V_p Δ) A_p .

(Both bounds are checked on exact arcs by second differences in `test_hddot_bounds_hold_numerically`.)

**Lemma 1.** Let `u` be held on `[t_k, t_k + Δ]`, let `|ḧ| ≤ M` on that interval, `α Δ ≤ 1`,
`h(t_k) ≥ 0`, and impose at `t_k`

    ḣ(t_k) + α h(t_k) ≥ δ,    δ = M Δ / 2 .

Then for every `s ∈ [0, Δ]`: `h(t_k + s) ≥ h(t_k)(1 − α s) ≥ 0`.

*Proof.* `ḣ(t_k + s) ≥ ḣ(t_k) − M s`. Integrating,
`h(t_k + s) ≥ h_0 + s ḣ(t_k) − M s²/2 ≥ h_0 + s(δ − α h_0) − M s²/2 = h_0(1 − α s) + (M s/2)(Δ − s)`.
The last term is `≥ 0` for `s ≤ Δ`, and `1 − α s ≥ 0` for `s ≤ Δ ≤ 1/α`. ∎

By induction over periods, `h ≥ 0` at every sample and between samples, provided every
constraint is feasible at every period and `h(0) ≥ 0`.

**Implementation.** Obstacle row: `g_o·u ≥ −α h_o + δ_o`, `δ_o = (Δ/2) M_o(d_k)`.
Robot–robot, decentralised and equal split: robot `i` enforces
`g_i·u_i ≥ −(α/2) h_ij + δ_ij/2`, `δ_ij = (Δ/2) M_r(d_k)`; the two rows sum to
`ḣ_ij + α h_ij ≥ δ_ij`, which is the lemma's hypothesis for the pair. This needs both robots to
solve their QP in the same period and use the same state (A3).

**Range rule.** A barrier is left out of the QP only when it cannot reach zero within the
period: obstacle `|p − c| > sensing_range ≥ D_o + V_p Δ`; neighbour
`|p_i − p_j| > comm_range ≥ D_rr + 2 V_p Δ`. `FilterConfig.validate()` enforces both, and also
`αΔ ≤ 1`, `ℓ > 0`, and `v_min ≤ 0 ≤ v_max`.

**Conservatism of the margin.** Since `δ > 0`, a robot at rest is no longer "safe enough" when
`h < δ/α`: the constraint then demands motion away. The resulting stand-off of the barrier
points for the placeholder configuration (`Δ = 0.1`, `V = 0.5`, `ω_max = 1`, `ℓ = 0.05`):

| α | extra obstacle stand-off beyond `D_o` | extra robot–robot stand-off beyond `D_rr` |
|---|---|---|
| 0.5 | 12.5 cm | 36.8 cm |
| 2 | 3.2 cm | 10.7 cm |
| 5 | 1.3 cm | 4.5 cm |
| 10 | 0.65 cm | 2.3 cm |

Larger `α` gives a smaller margin but a stiffer filter (more abrupt interventions, a larger
share of infeasible steps near the limit of `αΔ ≤ 1`). The barrier-gain sweep in the study
measures exactly this trade-off.

## 5. The QP and its solver

For each robot, every step:

    minimise   ½ |u − u_nom|²
    subject to g_o·u ≥ −α h_o + δ_o                for every obstacle within sensing range
               g_i·u ≥ −(α/2) h_ij + δ_ij/2        for every neighbour within comm range
               v_min ≤ v ≤ v_max,   |ω| ≤ ω_max     (the study's velocity bounds)

`u_nom` is the frozen policy's deterministic action. The solver (`qp2d.py`) is exact: the nearest
point of a convex polygon to `u_nom` is `u_nom`, a projection onto one constraint line, or a
vertex, so it enumerates those candidates and keeps the nearest feasible one. It has no iteration
limit, so "infeasible" means the constraints are contradictory (tolerance `1e-9` in
m/s and rad/s), not that a solver stopped. OSQP is used only in the tests as an independent
cross-check (`test_matches_osqp_on_feasible_problems`, `test_infeasible_detected_and_agrees_with_osqp`).
With no active barrier the QP returns `clip(u_nom)`, which is exactly what the filter-off
condition applies.

**Infeasibility is a reported outcome, never repaired.** If any robot's QP is infeasible the trial
stops before any action is applied and is counted as a failure (`INFEASIBLE`). The code does not
fall back to braking or to the nominal command, so infeasible steps cannot be hidden inside
"filter on" results. Two constructed cases are tested: a robot pinched between two obstacles
(rows contradict because `g_f = −g_b` and `δ > 0`), and an obstacle straight ahead with
`v_min = 0` (escape needs `v < 0`).

## 6. Evaluation protocol (identical for filter-on and filter-off)

1. **Motion between control steps is scored, not just end points.** One period is an exact
   unicycle arc. Robot–obstacle and robot–robot separation are checked on 64 sub-samples with the
   Lipschitz correction `true min separation ≥ sampled min − L τ/2` (`L = |v|` or `|v_i| + |v_j|`,
   `τ = Δ/64`). A period is scored as a collision when that lower bound is `< 0`, or when the
   simulator's realised end poses overlap. The rule never misses a true overlap; it can flag a
   near miss closer than `L τ/2` (under 1 mm for the placeholder settings), and it flags exact
   touching. Constructed crossings (`tests/test_geometry.py`, `tests/test_protocol.py`) show a
   robot whose sampled poses straddle an obstacle without overlap being caught.
2. **Stop at the first geometric collision.** Collision ends the trial, so a robot can never pass
   through an obstacle and later "complete" the task (`test_stop_at_first_collision_not_pass_through_and_complete`).
3. **Distinct outcomes:** `GOAL`, `COLLISION`, `INFEASIBLE`, `TIMEOUT`. Success is `GOAL` only;
   collision, infeasibility and timeout are task failures. Precedence inside one period:
   infeasible (decided before motion) > collision > goal > timeout, and collision beats goal if
   both occur in the same period.
4. **Same code path:** the only difference between conditions is whether the raw action goes
   through the filter or through `clip` (`test_identical_rules_when_filter_is_inactive` verifies
   identical trajectories when the filter has nothing to do).
5. **Logged per step and robot:** raw nominal action, clipped nominal action, applied action,
   minimum true obstacle and robot–robot barrier values, intervention flag, solver status
   (`nominal / modified / infeasible`), and infeasible flag; per trial: outcome, end step,
   collision kind, intervention and infeasible counts, minimum barrier values, final formation
   error, centroid-to-goal distance, stalled flag, and the largest gap between the simulator's
   end pose and the exact arc.

## 7. Assumptions

| # | Assumption | Where it matters |
|---|---|---|
| A1 | Realised motion is the exact unicycle arc for the held `(v, ω)`. Checked in the gate test for VMAS with drag 0, RK4, `collide=False`; the protocol logs the per-period gap (float32 level, below 1e-4 in tests). | Lemma 1 uses the exact arc |
| A2 | Exact, noise-free state and obstacle positions | `h` is evaluated on the true state |
| A3 | Synchronous, undelayed neighbour states within `comm_range` | equal-split robot–robot rows |
| A4 | Commands applied at the control period `Δ` with ZOH, QP solved within the period | Lemma 1 |
| A5 | `h(0) ≥ 0` for every barrier (the layout sampler enforces it) | invariance starts inside the set |
| A6 | Static circular obstacles, no arena boundary | no wall rows exist |
| A7 | Velocity box contains `u = 0` and the same box is used by the policy clip and the QP | `δ`, stopping |
| A8 | `αΔ ≤ 1`, `ℓ > 0`, ranges as in the range rule | enforced by `validate()` |
| A9 | Every QP is feasible at every period | otherwise the trial ends as `INFEASIBLE` |

## 8. What the numerical tests do and do not show

They show: the code matches the derivation (Jacobian, footprint, margin bounds, Lemma 1 on 10^4-scale
random states including bang-bang adversarial nominal commands); the exact solver agrees with OSQP;
the scoring rules catch constructed between-step crossings; filter-on trials in an adversarial
sweep (policy drives at the nearest obstacle or neighbour; random box commands; several `α` and
`Δ`; kinematic and VMAS environments) produced no collisions and no negative barrier values
(4 200 paired trials, 3 378 filter-off collisions versus 0 filter-on collisions, smallest barrier
value 0.0022; `results/stress_summary.csv`).

They do not show: safety for arbitrary states, policies or parameters; safety under noise, delay,
actuation lag, non-circular obstacles or walls; absence of deadlock; absence of infeasibility
(in the 4 200-trial sweep, 232 filter-on trials ended `INFEASIBLE`: 231 of the 600 trials at
`α = 0.5`, where the margin is largest, and 1 of 2 400 at `α = 2`; none at `α = 5` or `10`.
Infeasibility is a measured outcome of the study, not an anomaly to be tuned away silently). The proof in section 4 is what
supports the sampled-data claim, and only under A1 to A9.
