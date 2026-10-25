"""
Per-trial recording and the metrics of the locked protocol (docs/PREFILTER_VALIDATION_PROTOCOL_LOCKED_*.md).

Executed periods. T_arm = TrialResult.end_step is the number of control periods actually executed:
GOAL / COLLISION end after the period in which they occur (that period was executed), INFEASIBLE
stops at the attempted period before any motion (not executed), TIMEOUT executes max_steps. The
step log's 'active' flag is NOT used: it is set before the infeasibility check.

RecordingEnv is a pass-through wrapper that stores the realised poses after reset (index 0) and
after every env.step (index t = period t). After a trial ends the protocol steps it with u = 0, so
indices t > T_arm are frozen copies; every metric here uses t <= T_arm only.

Common-prefix formation difference of a pair:  L = min(T_off, T_on);  for L > 0
    d = mean_{t=1..L} [ e_on(t) - e_off(t) ],   e(t) = protocol.formation_error at post-step state t.
The initial state (t = 0) is excluded; the collision-ending period is included; an infeasible
attempted period is excluded (it is not executed). L = 0 -> excluded with a reason, never zero.
"""
from __future__ import annotations

import numpy as np

from .cbf_qp import FilterConfig, barrier_values
from .protocol import Outcome, formation_error

REASON_OK = ""
REASON_OFF_ZERO = "off arm executed 0 periods"
REASON_ON_ZERO = "on arm executed 0 periods"
REASON_BOTH_ZERO = "both arms executed 0 periods"


class RecordingEnv:
    """Pass-through env wrapper: commands, poses and therefore scoring are unchanged."""

    def __init__(self, env):
        self.env, self.traj = env, []

    def reset(self, init_poses, obst_xy):
        self.env.reset(init_poses, obst_xy)
        self.traj = [self.env.poses()]

    def poses(self):
        return self.env.poses()

    def step(self, u):
        self.env.step(u)
        self.traj.append(self.env.poses())

    def trajectory(self, max_steps):
        """[max_steps + 1, B, N, 3]; periods the batch did not run are frozen copies of the last state."""
        P = np.stack(self.traj)
        if P.shape[0] < max_steps + 1:
            P = np.concatenate([P, np.repeat(P[-1:], max_steps + 1 - P.shape[0], axis=0)])
        return P


def executed_periods(res):
    return np.asarray(res.end_step, dtype=np.int64)


def formation_series(traj, slots):
    """[S+1, B] mean slot error at every recorded state."""
    return np.stack([formation_error(traj[t], slots) for t in range(traj.shape[0])])


def centroid_distance_series(traj, goal):
    return np.linalg.norm(traj[..., :2].mean(axis=2) - np.asarray(goal)[None], axis=2)


def common_prefix(e_off, T_off, e_on, T_on):
    """e_* [S+1, B] per-state formation error series (index 0 = initial state), T_* [B] executed
    periods. Returns d [B] (NaN where excluded), L [B], reason [B] (str, '' = eligible)."""
    e_off, e_on = np.asarray(e_off, float), np.asarray(e_on, float)
    T_off, T_on = np.asarray(T_off, int), np.asarray(T_on, int)
    L = np.minimum(T_off, T_on)
    B = len(L)
    d = np.full(B, np.nan)
    reason = np.full(B, REASON_OK, dtype=object)
    for b in range(B):
        if L[b] == 0:
            reason[b] = (REASON_BOTH_ZERO if T_off[b] == 0 and T_on[b] == 0 else
                         REASON_OFF_ZERO if T_off[b] == 0 else REASON_ON_ZERO)
            continue
        d[b] = float(np.mean(e_on[1:L[b] + 1, b] - e_off[1:L[b] + 1, b]))
    return d, L, reason


def full_trajectory_mean(e, T):
    """Mean of e(t) over t = 1..T per trial (all executed periods); NaN where T = 0."""
    e, T = np.asarray(e, float), np.asarray(T, int)
    return np.array([float(np.mean(e[1:T[b] + 1, b])) if T[b] > 0 else np.nan for b in range(len(T))])


def both_goal_full_trajectory(e_off, T_off, out_off, e_on, T_on, out_on):
    """Secondary: full-trajectory mean formation error in pairs where BOTH arms reach GOAL.
    Returns (mask [B], mean_off [B], mean_on [B]); values only where mask."""
    both = (np.asarray(out_off) == Outcome.GOAL) & (np.asarray(out_on) == Outcome.GOAL)
    m_off = np.where(both, full_trajectory_mean(e_off, T_off), np.nan)
    m_on = np.where(both, full_trajectory_mean(e_on, T_on), np.nan)
    return both, m_off, m_on


def reached_goal_region(cdist, T, outcome, radius=0.30):
    """Centroid within `radius` of the goal at some recorded state before any failure:
    t = 0..T for GOAL / TIMEOUT / INFEASIBLE, t = 0..T-1 for COLLISION (the collision period's end
    state is not before the collision). Does not change the outcome."""
    cdist, T, outcome = np.asarray(cdist, float), np.asarray(T, int), np.asarray(outcome)
    out = np.zeros(len(T), dtype=bool)
    for b in range(len(T)):
        last = T[b] - 1 if outcome[b] == Outcome.COLLISION else T[b]
        out[b] = last >= 0 and bool((cdist[:last + 1, b] <= radius).any())
    return out


def sampled_barrier_minima(traj, T, obst_xy, cfg: FilterConfig):
    """Minimum TRUE barrier values over the recorded control-instant states t = 0..T (terminal state
    included). A sampled diagnostic, NOT a certified continuous-path bound; between-sample motion is
    covered only by the swept-path collision scoring. Returns (h_obst_min [B], h_rr_min [B])."""
    T = np.asarray(T, int)
    B = len(T)
    ho, hr = np.full(B, np.inf), np.full(B, np.inf)
    for t in range(int(T.max()) + 1):
        h_o, h_r = barrier_values(cfg, traj[t], obst_xy)
        live = t <= T
        if h_o.size:
            ho = np.where(live, np.minimum(ho, h_o.min(axis=(1, 2))), ho)
        hr = np.where(live, np.minimum(hr, h_r.min(axis=(1, 2))), hr)
    return ho, hr
