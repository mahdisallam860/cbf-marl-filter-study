"""
B-match brake comparator (locked protocol D5). NOT a safety filter and solves no QP.

At each robot and control period:
  1. the frozen actor's command is clipped to the unchanged velocity box;
  2. the robot's enabled barrier rows are built exactly as CBFQPFilter builds them at the given
     gain: obstacle rows  g_o.u >= -alpha h_o + delta_o           (|p - c| <= sensing_range)
                          robot rows     g_i.u >= -(alpha/2) h_ij + delta_ij/2 (|p_i - p_j| <= comm_range)
  3. if the clipped command violates any enabled row -> apply (v, w) = (0, clipped w);
  4. otherwise apply the clipped command.

A row is violated when (b - a.u)/|a| > solver_tol, the normalised feasibility convention of
qp2d.solve_qp2d (a degenerate row |a| < 1e-12 is violated when b > solver_tol). The intervention
tolerance is not used. B-match is not claimed to satisfy the barrier rows: rotation moves the
offset point, and a neighbour can drive into a stopped robot.

BMatchAdapter runs B-match inside the unchanged protocol loop (clip path: the braked command is
inside the box, so the protocol's clip leaves it unchanged) and keeps its own log of the actor's
raw and clipped commands, the applied command and the brake flag. run_bmatch relabels the
protocol's step log so the braked command is never reported as the actor's output.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cbf_qp import FilterConfig, barrier_point, margin_obstacle, margin_robot, rot_jacobian
from .protocol import run_trials


def barrier_rows(cfg: FilterConfig, poses, obst_xy):
    """Barrier rows of CBFQPFilter (no velocity-box rows). Returns A [B,N,R,2], b [B,N,R],
    mask [B,N,R] with R = K + N (obstacles first, then neighbours; the self row is masked)."""
    poses = np.asarray(poses, dtype=np.float64)
    obst_xy = np.asarray(obst_xy, dtype=np.float64)
    B, N, _ = poses.shape
    K = obst_xy.shape[1]
    p = barrier_point(poses, cfg.ell)
    J = rot_jacobian(poses[..., 2], cfg.ell)
    A, b, m = [], [], []
    if K:
        q = p[:, :, None, :] - obst_xy[:, None, :, :]
        d = np.linalg.norm(q, axis=3)
        A.append(2.0 * np.einsum("bnji,bnkj->bnki", J, q))
        b.append(-cfg.alpha * (d ** 2 - cfg.d_obst ** 2) + margin_obstacle(cfg, d))
        m.append(d <= cfg.sensing_range)
    q = p[:, :, None, :] - p[:, None, :, :]
    d = np.linalg.norm(q, axis=3)
    eye = np.eye(N, dtype=bool)[None]
    A.append(2.0 * np.einsum("bnji,bnkj->bnki", J, q))
    b.append(-0.5 * cfg.alpha * (d ** 2 - cfg.d_rr ** 2) + 0.5 * margin_robot(cfg, d))
    m.append((d <= cfg.comm_range) & ~eye)
    return np.concatenate(A, axis=2), np.concatenate(b, axis=2), np.concatenate(m, axis=2)


def rows_violated(u, A, b, mask, tol):
    """[B,N] True where u [B,N,2] violates an enabled row by more than tol (qp2d normalisation)."""
    nrm = np.linalg.norm(A, axis=3)
    degenerate = mask & (nrm < 1e-12)
    ok = mask & ~degenerate
    safe = np.where(nrm < 1e-12, 1.0, nrm)
    viol = (b - np.einsum("bnrd,bnd->bnr", A, u)) / safe
    return ((viol > tol) & ok).any(axis=2) | (degenerate & (b > tol)).any(axis=2)


class BMatchBrake:
    def __init__(self, cfg: FilterConfig):
        cfg.validate()
        self.cfg = cfg

    def __call__(self, poses, u_nom, obst_xy):
        cfg = self.cfg
        u_nom = np.asarray(u_nom, dtype=np.float64)
        lo, hi = np.array([cfg.v_min, -cfg.w_max]), np.array([cfg.v_max, cfg.w_max])
        u_clip = np.clip(u_nom, lo, hi)
        A, b, mask = barrier_rows(cfg, poses, obst_xy)
        active = rows_violated(u_clip, A, b, mask, cfg.solver_tol)
        u = u_clip.copy()
        u[..., 0] = np.where(active, 0.0, u_clip[..., 0])
        return dict(u=u, u_nom_clip=u_clip, activated=active, n_rows=mask.sum(axis=2))


class BMatchAdapter:
    """policy(poses, layout) wrapper for protocol.run_trials (filter-off path). Call k of the
    protocol loop is control period k; the log covers every call, executed or not."""

    def __init__(self, policy, cfg: FilterConfig):
        self.policy, self.brake = policy, BMatchBrake(cfg)
        self.actor_raw, self.actor_clipped, self.applied, self.activated = [], [], [], []

    def __call__(self, poses, layout):
        raw = np.asarray(self.policy(poses, layout), dtype=np.float64)
        out = self.brake(poses, raw, layout.obst_xy)
        self.actor_raw.append(raw.copy())
        self.actor_clipped.append(out["u_nom_clip"])
        self.applied.append(out["u"])
        self.activated.append(out["activated"])
        return out["u"]


@dataclass
class BMatchRun:
    res: object                    # protocol.TrialResult; step_log relabelled (see run_bmatch)
    executed: np.ndarray           # [T,B] True where period k was executed (k < end_step)
    actor_raw: np.ndarray          # [T,B,N,2] the actor's own output, NaN where not executed
    actor_clipped: np.ndarray      # [T,B,N,2] actor output clipped to the box
    applied: np.ndarray            # [T,B,N,2] B-match command that was applied
    brake_active: np.ndarray       # [T,B,N] brake activation in an executed period
    n_brake_activations: np.ndarray  # [B] robot-periods with an activation
    command_source: str = "bmatch_no_qp"


def run_bmatch(env, policy, layout, pc, fc: FilterConfig, log_steps=True) -> BMatchRun:
    """One B-match condition through the unchanged protocol loop. fc carries the gain whose rows
    are used. Outcomes, scoring and termination are those of protocol.run_trials."""
    ad = BMatchAdapter(policy, fc)
    res = run_trials(env, ad, layout, pc, fc, use_filter=False, log_steps=log_steps)
    T, (B, N) = pc.max_steps, layout.init_poses.shape[:2]
    S = len(ad.applied)
    k = np.arange(T)[:, None]
    executed = k < res.end_step[None, :]                       # B-match never stops for infeasibility
    pad = lambda xs, shape, fill: np.concatenate([np.stack(xs), np.full((T - S,) + shape, fill)]) if S < T else np.stack(xs)
    ex4 = executed[:, :, None, None]
    actor_raw = np.where(ex4, pad(ad.actor_raw, (B, N, 2), np.nan), np.nan)
    actor_clip = np.where(ex4, pad(ad.actor_clipped, (B, N, 2), np.nan), np.nan)
    applied = np.where(ex4, pad(ad.applied, (B, N, 2), np.nan), np.nan)
    brake = executed[:, :, None] & pad(ad.activated, (B, N), False)
    if res.step_log is not None:
        L = res.step_log
        assert np.allclose(L["applied"][executed], applied[executed]), "protocol applied a different command"
        L["bmatch_command"] = L.pop("raw")                   # what the protocol saw as 'raw' is the brake output
        L.pop("clipped")
        L["raw"], L["clipped"] = actor_raw, actor_clip           # the actor's own output
        L["intervened"] = brake                                   # brake activation, executed periods only
        L["brake_active"] = brake
        L["status"] = np.where(executed[:, :, None], -1, -2)      # -1 'na': no QP was solved
    res.n_interventions = brake.sum(axis=(0, 2))
    return BMatchRun(res=res, executed=executed, actor_raw=actor_raw, actor_clipped=actor_clip, applied=applied,
                     brake_active=brake, n_brake_activations=brake.sum(axis=(0, 2)))
