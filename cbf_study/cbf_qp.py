"""
Execution-time CBF-QP safety filter for differential-drive (unicycle) robots.

The filter sits between a frozen policy's deterministic action and env.step(). It does not touch
the policy. Full derivation, assumptions and limits: docs/CBF_DESIGN.md.

Model (axle-centre pose x = (x, y, theta), body = disc of radius R about the axle centre):
    xdot = v cos(theta),  ydot = v sin(theta),  thetadot = w,       u = (v, w)

Barrier point  p = (x + l cos(theta), y + l sin(theta)),  l > 0 (offset ahead of the axle)
    pdot = J(theta) u,   J = [[cos th, -l sin th], [sin th, l cos th]] = Rot(th) diag(1, l)
    det J = l  ->  J is invertible, so every barrier below has RELATIVE DEGREE 1 in (v, w)
    (at l = 0 the point is the axle centre, J is singular, and the relative degree is not 1).

Footprint cover:  body disc(x, R) is inside disc(p, R + l) because |x - p| = l.
    robot-obstacle:  h_o  = |p - c|^2        - (R + l + R_o)^2        >= 0  =>  body clears obstacle
    robot-robot:     h_ij = |p_i - p_j|^2    - (2 (R + l))^2          >= 0  =>  bodies are disjoint

Constraint per barrier, held fixed over one control period dt (zero-order hold):
    hdot + alpha * h >= delta,       delta = (dt / 2) * M          (see CBF_DESIGN.md, Lemma 1)
with M a bound on |hddot| over the period. Robot-robot constraints are split equally:
    g_i . u_i >= -(alpha/2) h_ij + delta_ij / 2,   g_i = 2 J_i^T (p_i - p_j)
so that the two robots' rows sum to  hdot_ij + alpha h_ij >= delta_ij.

Velocity bounds v_min <= v <= v_max and |w| <= w_max are rows of the same QP.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .qp2d import STATUS_INFEASIBLE, STATUS_MODIFIED, STATUS_NOMINAL, solve_qp2d


@dataclass(frozen=True)
class FilterConfig:
    dt: float                 # control period [s]
    robot_r: float            # body radius R [m]
    obst_r: float             # obstacle radius R_o [m]
    ell: float                # barrier-point offset l [m], must be > 0
    v_min: float              # lower forward-speed bound [m/s]
    v_max: float              # upper forward-speed bound [m/s]
    w_max: float              # angular-speed bound |w| <= w_max [rad/s]
    alpha: float              # barrier gain [1/s]
    sensing_range: float      # obstacles with |p - c| <= this enter the QP [m]
    comm_range: float         # neighbours with |p_i - p_j| <= this enter the QP [m]
    intervention_tol: float = 1e-6   # |u_filtered - clip(u_nominal)|_inf above this = intervention
    solver_tol: float = 1e-9

    # ---- derived geometry and bounds -------------------------------------------------------
    @property
    def cover_r(self) -> float:
        return self.robot_r + self.ell

    @property
    def d_obst(self) -> float:
        return self.cover_r + self.obst_r

    @property
    def d_rr(self) -> float:
        return 2.0 * self.cover_r

    @property
    def v_abs_max(self) -> float:
        return max(abs(self.v_min), abs(self.v_max))

    @property
    def vp(self) -> float:
        """Upper bound on |pdot| over the control box."""
        return float(np.hypot(self.v_abs_max, self.ell * self.w_max))

    @property
    def ap(self) -> float:
        """Upper bound on |pddot| under constant (v, w): |w| |pdot| <= w_max * vp."""
        return self.w_max * self.vp

    def validate(self) -> None:
        if not self.ell > 0:
            raise ValueError("ell must be > 0 (relative degree 1 needs an invertible J)")
        if not (self.v_min <= 0.0 <= self.v_max):
            raise ValueError("v_min <= 0 <= v_max required so that stopping is allowed")
        if self.alpha * self.dt > 1.0 + 1e-12:
            raise ValueError("alpha * dt must be <= 1 (Lemma 1 hypothesis)")
        need_s = self.d_obst + self.vp * self.dt
        if self.sensing_range < need_s:
            raise ValueError(f"sensing_range {self.sensing_range} < d_obst + vp*dt = {need_s:.4f}")
        need_c = self.d_rr + 2.0 * self.vp * self.dt
        if self.comm_range < need_c:
            raise ValueError(f"comm_range {self.comm_range} < d_rr + 2*vp*dt = {need_c:.4f}")

    def to_dict(self):
        return asdict(self)


def rot_jacobian(theta, ell):
    """J(theta) stacked as [..., 2, 2]."""
    c, s = np.cos(theta), np.sin(theta)
    J = np.empty(theta.shape + (2, 2))
    J[..., 0, 0] = c
    J[..., 0, 1] = -ell * s
    J[..., 1, 0] = s
    J[..., 1, 1] = ell * c
    return J


def barrier_point(poses, ell):
    """poses [...,3] -> p [...,2]."""
    return np.stack([poses[..., 0] + ell * np.cos(poses[..., 2]),
                     poses[..., 1] + ell * np.sin(poses[..., 2])], axis=-1)


def margin_obstacle(cfg: FilterConfig, dist):
    """delta_o for a static obstacle at barrier-point distance `dist` at the start of the step.

    |hddot| <= 2 vp^2 + 2 (dist + vp dt) ap, since |p(t) - c| <= dist + vp dt inside the step.
    """
    M = 2.0 * cfg.vp ** 2 + 2.0 * (dist + cfg.vp * cfg.dt) * cfg.ap
    return 0.5 * cfg.dt * M


def margin_robot(cfg: FilterConfig, dist):
    """delta_ij for a robot pair at barrier-point distance `dist` at the start of the step.

    |hddot| <= 8 vp^2 + 4 (dist + 2 vp dt) ap  (relative speed <= 2 vp, relative accel <= 2 ap).
    """
    M = 8.0 * cfg.vp ** 2 + 4.0 * (dist + 2.0 * cfg.vp * cfg.dt) * cfg.ap
    return 0.5 * cfg.dt * M


def barrier_values(cfg: FilterConfig, poses, obst_xy):
    """True barrier values (no margin). poses [B,N,3], obst_xy [B,K,2].

    Returns h_obst [B,N,K], h_rr [B,N,N] (diagonal = +inf).
    """
    p = barrier_point(poses, cfg.ell)
    d_o = np.linalg.norm(p[:, :, None, :] - obst_xy[:, None, :, :], axis=3)
    h_o = d_o ** 2 - cfg.d_obst ** 2
    d_r = np.linalg.norm(p[:, :, None, :] - p[:, None, :, :], axis=3)
    h_r = d_r ** 2 - cfg.d_rr ** 2
    n = poses.shape[1]
    h_r[:, np.arange(n), np.arange(n)] = np.inf
    return h_o, h_r


class CBFQPFilter:
    """Decentralised CBF-QP: one 2-D QP per robot per step, solved exactly."""

    def __init__(self, cfg: FilterConfig):
        cfg.validate()
        self.cfg = cfg

    # -------------------------------------------------------------------------------------
    def __call__(self, poses, u_nom, obst_xy):
        """poses [B,N,3], u_nom [B,N,2] (raw policy action), obst_xy [B,K,2].

        Returns a dict of arrays (all float64 / int64 / bool):
          u            [B,N,2]  command to apply (NaN where infeasible)
          u_nom_clip   [B,N,2]  nominal command clipped to the velocity box
          status       [B,N]    0 nominal feasible, 1 modified, 2 infeasible
          infeasible   [B,N]    status == 2
          intervened   [B,N]    status != 2 and |u - u_nom_clip|_inf > intervention_tol
          h_obst_min   [B,N]    min true obstacle barrier value over ALL obstacles (inf if none)
          h_rr_min     [B,N]    min true robot-robot barrier value over ALL neighbours
          n_rows       [B,N]    number of enabled barrier rows (obstacle + neighbour)
          dist_to_nom  [B,N]    |u - u_nom_clip|_2
        """
        cfg = self.cfg
        poses = np.asarray(poses, dtype=np.float64)
        u_nom = np.asarray(u_nom, dtype=np.float64)
        obst_xy = np.asarray(obst_xy, dtype=np.float64)
        B, N, _ = poses.shape
        K = obst_xy.shape[1]

        p = barrier_point(poses, cfg.ell)                                    # [B,N,2]
        J = rot_jacobian(poses[..., 2], cfg.ell)                             # [B,N,2,2]

        rows_A, rows_b, rows_m = [], [], []

        # --- obstacle rows: g = 2 J^T (p - c),  b = -alpha h + delta_o -----------------------
        if K:
            q = p[:, :, None, :] - obst_xy[:, None, :, :]                    # [B,N,K,2]
            d = np.linalg.norm(q, axis=3)
            h = d ** 2 - cfg.d_obst ** 2
            g = 2.0 * np.einsum("bnji,bnkj->bnki", J, q)                     # J^T q
            rows_A.append(g)
            rows_b.append(-cfg.alpha * h + margin_obstacle(cfg, d))
            rows_m.append(d <= cfg.sensing_range)
            h_obst_min = h.min(axis=2)
        else:
            h_obst_min = np.full((B, N), np.inf)

        # --- robot-robot rows (equal split) ---------------------------------------------------
        if N > 1:
            q = p[:, :, None, :] - p[:, None, :, :]                          # [B,N,N,2] (i minus j)
            d = np.linalg.norm(q, axis=3)
            h = d ** 2 - cfg.d_rr ** 2
            g = 2.0 * np.einsum("bnji,bnkj->bnki", J, q)                     # row i uses J_i
            eye = np.eye(N, dtype=bool)[None]
            rows_A.append(g)
            rows_b.append(-0.5 * cfg.alpha * h + 0.5 * margin_robot(cfg, d))
            rows_m.append((d <= cfg.comm_range) & ~eye)
            h_rr = np.where(eye, np.inf, h)
            h_rr_min = h_rr.min(axis=2)
        else:
            h_rr_min = np.full((B, N), np.inf)

        # --- velocity box rows ----------------------------------------------------------------
        box_A = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        box_b = np.array([cfg.v_min, -cfg.v_max, -cfg.w_max, -cfg.w_max])
        rows_A.append(np.broadcast_to(box_A, (B, N, 4, 2)))
        rows_b.append(np.broadcast_to(box_b, (B, N, 4)))
        rows_m.append(np.ones((B, N, 4), dtype=bool))

        A = np.concatenate(rows_A, axis=2)
        bvec = np.concatenate(rows_b, axis=2)
        mask = np.concatenate(rows_m, axis=2)
        K_tot = A.shape[2]

        lo = np.array([cfg.v_min, -cfg.w_max])
        hi = np.array([cfg.v_max, cfg.w_max])
        u_clip = np.clip(u_nom, lo, hi)

        u_flat, status = solve_qp2d(u_nom.reshape(B * N, 2), A.reshape(B * N, K_tot, 2),
                                    bvec.reshape(B * N, K_tot), mask.reshape(B * N, K_tot),
                                    tol=cfg.solver_tol)
        u = u_flat.reshape(B, N, 2)
        status = status.reshape(B, N)
        infeasible = status == STATUS_INFEASIBLE
        dist = np.linalg.norm(u - u_clip, axis=2)
        intervened = ~infeasible & (np.abs(u - u_clip).max(axis=2) > cfg.intervention_tol)
        n_rows = mask[:, :, :K_tot - 4].sum(axis=2)
        return dict(u=u, u_nom_clip=u_clip, status=status, infeasible=infeasible,
                    intervened=intervened, h_obst_min=h_obst_min, h_rr_min=h_rr_min,
                    n_rows=n_rows, dist_to_nom=dist)
