"""
Trial protocol shared by filter-on and filter-off runs.

The ONLY difference between the two conditions is one line: whether the raw policy action goes
through CBFQPFilter or is just clipped to the velocity box. Collision scoring, termination, goal
test, outcome precedence and logging are the same code path.

Per control step k (all trials in the batch advance together, finished trials are frozen):
  1. raw = policy(poses, layout)                      deterministic, frozen
  2. filter on : u = CBFQPFilter(poses, raw, obstacles)
     filter off: u = clip(raw, velocity box)
  3. if any robot's QP is infeasible -> outcome INFEASIBLE; the trial stops BEFORE any action is
     applied (the state is left unchanged and no further step is taken)
  4. otherwise apply u for one control period (zero-order hold) and score the MOTION of that
     period: exact unicycle arcs checked with the conservative swept rule (geometry.py), plus the
     environment's realised end poses. First geometric overlap -> outcome COLLISION; the trial
     stops there, so a robot can never pass through an obstacle and later "complete" the task.
  5. goal test on the realised poses (centroid within goal_tol and formation error within
     formation_tol); reaching it without a collision in that period -> outcome GOAL
  6. after max_steps with none of the above -> outcome TIMEOUT

Outcomes are distinct. Success = GOAL only. COLLISION and INFEASIBLE are task failures.
Precedence inside one period: INFEASIBLE (decided before motion) > COLLISION > GOAL > TIMEOUT,
and COLLISION beats GOAL if both happen in the same period.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from enum import IntEnum

import numpy as np

from .cbf_qp import CBFQPFilter, FilterConfig, barrier_values
from .geometry import DEFAULT_SUBSTEPS, arc_poses, swept_sep_lb_obstacle, swept_sep_lb_robots


class Outcome(IntEnum):
    RUNNING = 0
    GOAL = 1
    COLLISION = 2
    INFEASIBLE = 3
    TIMEOUT = 4


FAILURES = (Outcome.COLLISION, Outcome.INFEASIBLE, Outcome.TIMEOUT)
STATUS_NAMES = {-1: "na", 0: "nominal", 1: "modified", 2: "infeasible"}


@dataclass(frozen=True)
class ProtocolConfig:
    dt: float
    robot_r: float
    obst_r: float
    v_min: float
    v_max: float
    w_max: float
    max_steps: int
    goal_tol: float             # centroid-to-goal distance [m]
    formation_tol: float        # mean slot error at the end [m]
    substeps: int = DEFAULT_SUBSTEPS
    stall_window: int = 50      # deadlock diagnostic: centroid progress over this many steps ...
    stall_dist: float = 0.05    # ... below this distance flags a stalled TIMEOUT


def check_consistent(pc: ProtocolConfig, fc: FilterConfig):
    same = (pc.dt == fc.dt and pc.robot_r == fc.robot_r and pc.obst_r == fc.obst_r and
            pc.v_min == fc.v_min and pc.v_max == fc.v_max and pc.w_max == fc.w_max)
    if not same:
        raise ValueError("ProtocolConfig and FilterConfig must share dt, radii and velocity bounds")


def formation_error(poses, slots):
    """Mean distance between each robot's centroid-relative position and its slot. [B]"""
    xy = poses[..., :2]
    rel = xy - xy.mean(axis=1, keepdims=True)
    return np.linalg.norm(rel - slots[None], axis=2).mean(axis=1)


def goal_reached(poses, layout, pc):
    cen = poses[..., :2].mean(axis=1)
    near = np.linalg.norm(cen - layout.goal, axis=1) <= pc.goal_tol
    return near & (formation_error(poses, layout.slots) <= pc.formation_tol)


def score_motion(pose0, u, pose1, obst_xy, pc: ProtocolConfig):
    """Geometric overlap during one control period.

    pose0/pose1 [B,N,3] before/after, u [B,N,2] applied. Returns (hit_obst [B], hit_rr [B],
    track_err [B]) where track_err is the largest gap between the realised end pose and the exact
    arc end (a validity check on the simulator, not used for scoring).
    """
    B, N, _ = pose0.shape
    K = obst_xy.shape[1]
    flat0 = pose0.reshape(B * N, 3)
    v = u[..., 0].reshape(-1)
    w = u[..., 1].reshape(-1)
    hit_o = np.zeros(B, dtype=bool)
    r_o = pc.robot_r + pc.obst_r
    for k in range(K):
        c = np.repeat(obst_xy[:, k, :], N, axis=0)
        lb = swept_sep_lb_obstacle(flat0, v, w, pc.dt, c, r_o, pc.substeps).reshape(B, N)
        end = np.linalg.norm(pose1[..., :2] - obst_xy[:, k, None, :], axis=2) - r_o
        hit_o |= ((lb < 0) | (end < 0)).any(axis=1)
    hit_r = np.zeros(B, dtype=bool)
    for i in range(N):
        for j in range(i + 1, N):
            lb = swept_sep_lb_robots(pose0[:, i], u[:, i, 0], u[:, i, 1],
                                     pose0[:, j], u[:, j, 0], u[:, j, 1],
                                     pc.dt, 2 * pc.robot_r, pc.substeps)
            end = np.linalg.norm(pose1[:, i, :2] - pose1[:, j, :2], axis=1) - 2 * pc.robot_r
            hit_r |= (lb < 0) | (end < 0)
    arc_end = arc_poses(flat0, v, w, pc.dt, 1)[:, -1, :].reshape(B, N, 3)
    dth = (pose1[..., 2] - arc_end[..., 2] + np.pi) % (2 * np.pi) - np.pi
    err = np.maximum(np.linalg.norm(pose1[..., :2] - arc_end[..., :2], axis=2), np.abs(dth)).max(axis=1)
    return hit_o, hit_r, err


@dataclass
class TrialResult:
    use_filter: bool
    outcome: np.ndarray            # [B] Outcome codes
    end_step: np.ndarray           # [B] control periods applied before the trial ended
    collision_kind: np.ndarray     # [B] 0 none, 1 robot-obstacle, 2 robot-robot
    final_poses: np.ndarray        # [B,N,3]
    form_err_final: np.ndarray     # [B]
    form_err_mean: np.ndarray      # [B]
    goal_dist_final: np.ndarray    # [B]
    n_interventions: np.ndarray    # [B] robot-steps with a modified command
    n_infeasible: np.ndarray       # [B] robot-steps with an infeasible QP (counted before stopping)
    h_obst_min: np.ndarray         # [B] min true obstacle barrier value seen during the trial
    h_rr_min: np.ndarray           # [B] min true robot-robot barrier value seen
    stalled: np.ndarray            # [B] TIMEOUT with centroid progress < stall_dist over the window
    track_err_max: np.ndarray      # [B] max simulator-vs-exact-arc gap
    step_log: dict | None = None   # arrays [T,B,N,...], NaN/-1 where the trial was already over

    @property
    def success(self):
        return self.outcome == Outcome.GOAL


def run_trials(env, policy, layout, pc: ProtocolConfig, fc: FilterConfig, use_filter: bool,
               log_steps: bool = True) -> TrialResult:
    check_consistent(pc, fc)
    filt = CBFQPFilter(fc) if use_filter else None
    env.reset(layout.init_poses, layout.obst_xy)
    obst = np.asarray(layout.obst_xy, dtype=np.float64)
    B, N, _ = layout.init_poses.shape
    T = pc.max_steps
    lo = np.array([pc.v_min, -pc.w_max])
    hi = np.array([pc.v_max, pc.w_max])

    outcome = np.zeros(B, dtype=np.int64)
    end_step = np.full(B, T, dtype=np.int64)
    ckind = np.zeros(B, dtype=np.int64)
    n_int = np.zeros(B, dtype=np.int64)
    n_inf = np.zeros(B, dtype=np.int64)
    h_o_min = np.full(B, np.inf)
    h_r_min = np.full(B, np.inf)
    track = np.zeros(B)
    form_sum = np.zeros(B)
    form_cnt = np.zeros(B)
    cen_dist = np.zeros((T + 1, B))
    poses = env.poses()
    final = poses.copy()
    log = None
    if log_steps:
        nan = lambda *s: np.full((T,) + s, np.nan)
        log = dict(raw=nan(B, N, 2), clipped=nan(B, N, 2), applied=nan(B, N, 2),
                   h_obst_min=nan(B, N), h_rr_min=nan(B, N),
                   status=np.full((T, B, N), -2, dtype=np.int64),
                   intervened=np.zeros((T, B, N), dtype=bool),
                   infeasible=np.zeros((T, B, N), dtype=bool),
                   active=np.zeros((T, B), dtype=bool))

    # ---- time 0: an initial overlap is a collision; an initial goal is a goal ------------------
    h0, hh0 = barrier_values(fc, poses, obst)   # diagnostics only
    cen_dist[0] = np.linalg.norm(poses[..., :2].mean(axis=1) - layout.goal, axis=1)
    r_o = pc.robot_r + pc.obst_r
    d_o0 = np.linalg.norm(poses[:, :, None, :2] - obst[:, None, :, :], axis=3) - r_o if obst.shape[1] else np.full((B, N, 1), 1.0)
    d_r0 = np.linalg.norm(poses[:, :, None, :2] - poses[:, None, :, :2], axis=3) - 2 * pc.robot_r + np.eye(N)[None] * 1e9
    c0_o = (d_o0 < 0).any(axis=(1, 2))
    c0_r = (d_r0 < 0).any(axis=(1, 2))
    outcome[c0_o | c0_r] = Outcome.COLLISION
    ckind[c0_o] = 1
    ckind[~c0_o & c0_r] = 2
    end_step[outcome != 0] = 0
    g0 = goal_reached(poses, layout, pc) & (outcome == 0)
    outcome[g0] = Outcome.GOAL
    end_step[g0] = 0

    for k in range(T):
        act = outcome == Outcome.RUNNING
        if not act.any():
            break
        raw = np.asarray(policy(poses, layout), dtype=np.float64)
        clipped = np.clip(raw, lo, hi)
        if use_filter:
            fo = filt(poses, raw, obst)
            u, status = fo["u"], fo["status"]
            infeas, interv = fo["infeasible"], fo["intervened"]
            h_o, h_r = fo["h_obst_min"], fo["h_rr_min"]
        else:
            u, status = clipped, np.full((B, N), -1, dtype=np.int64)
            infeas = np.zeros((B, N), dtype=bool)
            interv = np.zeros((B, N), dtype=bool)
            ho_all, hr_all = barrier_values(fc, poses, obst)
            h_o = ho_all.min(axis=2) if obst.shape[1] else np.full((B, N), np.inf)
            h_r = hr_all.min(axis=2)

        h_o_min = np.where(act, np.minimum(h_o_min, h_o.min(axis=1)), h_o_min)
        h_r_min = np.where(act, np.minimum(h_r_min, h_r.min(axis=1)), h_r_min)
        n_int += np.where(act, interv.sum(axis=1), 0)
        n_inf += np.where(act, infeas.sum(axis=1), 0)
        if log is not None:
            a3 = act[:, None]
            log["raw"][k] = np.where(a3[..., None], raw, np.nan)
            log["clipped"][k] = np.where(a3[..., None], clipped, np.nan)
            log["applied"][k] = np.where(a3[..., None], u, np.nan)
            log["h_obst_min"][k] = np.where(a3, h_o, np.nan)
            log["h_rr_min"][k] = np.where(a3, h_r, np.nan)
            log["status"][k] = np.where(a3, status, -2)
            log["intervened"][k] = interv & a3
            log["infeasible"][k] = infeas & a3
            log["active"][k] = act

        # 3. infeasible QP: stop before moving
        inf_env = act & infeas.any(axis=1)
        outcome[inf_env] = Outcome.INFEASIBLE
        end_step[inf_env] = k
        final[inf_env] = poses[inf_env]
        go = act & ~inf_env

        u_apply = np.where(go[:, None, None], np.nan_to_num(u), 0.0)
        env.step(u_apply)
        new = env.poses()

        # 4. score the motion of this period
        hit_o, hit_r, err = score_motion(poses, u_apply, new, obst, pc)
        track = np.where(go, np.maximum(track, err), track)
        hit = go & (hit_o | hit_r)
        outcome[hit] = Outcome.COLLISION
        ckind[hit & hit_o] = 1
        ckind[hit & ~hit_o & hit_r] = 2
        end_step[hit] = k + 1
        final[hit] = new[hit]

        # 5. goal test (only trials still running)
        still = go & ~hit
        ferr = formation_error(new, layout.slots)
        form_sum += np.where(still, ferr, 0.0)
        form_cnt += still
        cen_dist[k + 1] = np.linalg.norm(new[..., :2].mean(axis=1) - layout.goal, axis=1)
        gl = still & goal_reached(new, layout, pc)
        outcome[gl] = Outcome.GOAL
        end_step[gl] = k + 1
        final[gl] = new[gl]
        poses = new

    run_out = outcome == Outcome.RUNNING
    outcome[run_out] = Outcome.TIMEOUT
    final[run_out] = poses[run_out]
    end_step[run_out] = T

    idx = np.arange(B)
    lag = np.clip(end_step - pc.stall_window, 0, None)
    progress = cen_dist[lag, idx] - cen_dist[np.minimum(end_step, T), idx]
    stalled = (outcome == Outcome.TIMEOUT) & (progress < pc.stall_dist)
    return TrialResult(
        use_filter=use_filter, outcome=outcome, end_step=end_step, collision_kind=ckind,
        final_poses=final, form_err_final=formation_error(final, layout.slots),
        form_err_mean=form_sum / np.maximum(form_cnt, 1),
        goal_dist_final=np.linalg.norm(final[..., :2].mean(axis=1) - layout.goal, axis=1),
        n_interventions=n_int, n_infeasible=n_inf, h_obst_min=h_o_min, h_rr_min=h_r_min,
        stalled=stalled, track_err_max=track, step_log=log)


def run_paired(env_factory, policy, layout, pc, fc, log_steps=True):
    """Same layout, same policy, same rules; filter off then filter on."""
    off = run_trials(env_factory(), policy, layout, pc, fc, use_filter=False, log_steps=log_steps)
    on = run_trials(env_factory(), policy, layout, pc, fc, use_filter=True, log_steps=log_steps)
    return off, on


# --------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------
def write_summary_csv(path, res: TrialResult, label=""):
    cols = ["label", "trial", "filter", "outcome", "success", "end_step", "collision_kind",
            "form_err_final", "form_err_mean", "goal_dist_final", "n_interventions",
            "n_infeasible", "h_obst_min", "h_rr_min", "stalled", "track_err_max"]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for b in range(len(res.outcome)):
            wr.writerow([label, b, int(res.use_filter), Outcome(res.outcome[b]).name,
                         int(res.success[b]), res.end_step[b], res.collision_kind[b],
                         f"{res.form_err_final[b]:.6f}", f"{res.form_err_mean[b]:.6f}",
                         f"{res.goal_dist_final[b]:.6f}", res.n_interventions[b], res.n_infeasible[b],
                         f"{res.h_obst_min[b]:.6g}", f"{res.h_rr_min[b]:.6g}",
                         int(res.stalled[b]), f"{res.track_err_max[b]:.3e}"])


def write_step_csv(path, res: TrialResult, trials=None, label=""):
    """Long-format per-step, per-robot log: nominal, clipped nominal, applied action, barrier
    values, intervention flag, solver status, infeasibility flag."""
    L = res.step_log
    if L is None:
        raise ValueError("run with log_steps=True")
    trials = range(L["active"].shape[1]) if trials is None else trials
    cols = ["label", "trial", "step", "robot", "filter", "nom_v", "nom_w", "clip_v", "clip_w",
            "app_v", "app_w", "h_obst_min", "h_rr_min", "intervened", "status", "infeasible"]
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for b in trials:
            for k in range(L["active"].shape[0]):
                if not L["active"][k, b]:
                    continue
                for i in range(L["raw"].shape[2]):
                    wr.writerow([label, b, k, i, int(res.use_filter),
                                 *(f"{x:.6g}" for x in L["raw"][k, b, i]),
                                 *(f"{x:.6g}" for x in L["clipped"][k, b, i]),
                                 *(f"{x:.6g}" for x in L["applied"][k, b, i]),
                                 f"{L['h_obst_min'][k, b, i]:.6g}", f"{L['h_rr_min'][k, b, i]:.6g}",
                                 int(L["intervened"][k, b, i]), STATUS_NAMES[int(L["status"][k, b, i])],
                                 int(L["infeasible"][k, b, i])])
