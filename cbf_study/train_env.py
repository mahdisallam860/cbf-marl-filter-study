"""
VMAS formation-navigation environment for MAPPO training (filter OFF).

It matches the evaluation environment by construction rather than by copy:
  * physics: TrainVmasEnv IS envs.VmasEnv (same scenario class: DiffDrive, world drag 0, RK4,
    collide=False, same dt, radii and velocity bounds); it only adds a per-environment reset that
    leaves the other environments' VMAS state untouched;
  * layouts: scenario.sample_layout with the study config, on training seeds that never coincide
    with validation or pilot seeds (config.train_seed);
  * commands: raw policy action clipped to the same velocity box the protocol uses;
  * collisions: never taken from the engine; each period is scored with protocol.score_motion
    (exact arcs, swept rule, realised end poses) and the episode stops at the first collision;
  * goal: protocol.goal_reached; timeout after max_steps periods (a truncation, bootstrapped).

Reward per period (weights from [reward] in config/study.toml):
    w_progress  * (|c_k - goal| - |c_{k+1} - goal|)          centroid progress
  - w_formation * formation_error(poses_{k+1})                 mean slot error (protocol definition)
  - w_control   * mean_i[(v_i / v_abs_max)^2 + (w_i / w_max)^2]
  - w_collision_obst * [robot-obstacle collision in the period]
  - w_collision_rr   * [robot-robot collision in the period]
  - w_proximity * sum over robot-obstacle and robot-robot pairs of max(0, 1 - clearance / proximity_margin)
    (config v2; clearance = surface gap at the end of the period; shaping only, collisions are still
    scored by protocol.score_motion)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import StudyConfig
from .envs import VmasEnv
from .observations import build_obs
from .protocol import Outcome, formation_error, goal_reached, score_motion
from .scenario import Layout, sample_layout


def proximity_penalty(poses, obst_xy, pc, margin):
    """sum over robot-obstacle and robot-robot pairs of max(0, 1 - surface clearance / margin). [B]"""
    xy = poses[..., :2]
    pen = np.zeros(xy.shape[0])
    if obst_xy.shape[1]:
        clr = np.linalg.norm(xy[:, :, None] - obst_xy[:, None], axis=3) - (pc.robot_r + pc.obst_r)
        pen += np.clip(1.0 - clr / margin, 0.0, None).sum(axis=(1, 2))
    n = xy.shape[1]
    i, j = np.triu_indices(n, 1)
    clr = np.linalg.norm(xy[:, i] - xy[:, j], axis=2) - 2 * pc.robot_r
    pen += np.clip(1.0 - clr / margin, 0.0, None).sum(axis=1)
    return pen


class TrainVmasEnv(VmasEnv):
    """envs.VmasEnv plus a reset of selected batch entries."""

    def reset_subset(self, mask, init_poses, obst_xy):
        torch = self._torch
        m = torch.as_tensor(np.asarray(mask, dtype=bool))[:, None]
        init_poses = np.asarray(init_poses)
        for i, a in enumerate(self.env.agents):
            p = torch.as_tensor(init_poses[:, i, :], dtype=torch.float32)
            a.set_pos(torch.where(m, p[:, :2], a.state.pos), batch_index=None)
            a.set_rot(torch.where(m, p[:, 2:3], a.state.rot), batch_index=None)
            a.set_vel(torch.where(m, torch.zeros_like(a.state.vel), a.state.vel), batch_index=None)
            a.set_ang_vel(torch.where(m, torch.zeros_like(a.state.ang_vel), a.state.ang_vel), batch_index=None)
        obst_xy = np.asarray(obst_xy)
        for j, lm in enumerate(self.env.world.landmarks):
            c = torch.as_tensor(obst_xy[:, j, :], dtype=torch.float32)
            lm.set_pos(torch.where(m, c, lm.state.pos), batch_index=None)


class LayoutStream:
    """Fresh training layouts for one density, drawn chunk by chunk from scenario.sample_layout."""

    def __init__(self, cfg: StudyConfig, run_seed: int, n_obst: int, chunk: int = 256):
        self.cfg, self.run_seed, self.n_obst, self.chunk = cfg, run_seed, n_obst, chunk
        self.fc, self.sc = cfg.layout_filter_config(), cfg.scenario_config()
        self.idx = 0
        self.buf = None
        self.pos = 0
        self.drawn = 0

    def _refill(self):
        seed = self.cfg.train_seed(self.run_seed, self.n_obst, self.idx)
        self.buf = sample_layout(seed, self.chunk, self.n_obst, self.fc, self.sc)
        self.idx += 1
        self.pos = 0

    def take(self, n):
        inits, obsts, goals = [], [], []
        while n:
            if self.buf is None or self.pos == self.chunk:
                self._refill()
            m = min(n, self.chunk - self.pos)
            sl = slice(self.pos, self.pos + m)
            inits.append(self.buf.init_poses[sl]); obsts.append(self.buf.obst_xy[sl]); goals.append(self.buf.goal[sl])
            self.pos += m
            n -= m
        self.drawn += sum(len(g) for g in goals)
        return np.concatenate(inits), np.concatenate(obsts), np.concatenate(goals), self.buf.slots


@dataclass
class StepOut:
    obs: np.ndarray            # [E,N,D] observation after auto-reset
    reward: np.ndarray         # [E]
    terminated: np.ndarray     # [E] GOAL or COLLISION
    truncated: np.ndarray      # [E] TIMEOUT
    final_obs: np.ndarray      # [E,N,D] observation of the end state before reset (for bootstrapping)
    episodes: list             # finished episodes: dicts with density, outcome, length, return, form_err


class _Group:
    def __init__(self, cfg, run_seed, n_obst, n_envs, vmas_seed):
        self.cfg, self.K, self.E = cfg, n_obst, n_envs
        fc = cfg.layout_filter_config()
        self.env = TrainVmasEnv(n_envs, cfg.scenario_config().n_agents, n_obst, cfg.dt, fc.robot_r, fc.obst_r,
                                fc.v_max, fc.w_max, seed=vmas_seed)
        self.stream = LayoutStream(cfg, run_seed, n_obst)
        self.layout = None
        self.poses = None
        self.t = np.zeros(n_envs, dtype=np.int64)
        self.ret = np.zeros(n_envs)
        self.form_sum = np.zeros(n_envs)

    def reset(self, layout: Layout | None = None):
        if layout is None:
            init, obst, goal, slots = self.stream.take(self.E)
            layout = Layout(init, obst, goal, slots)
        self.layout = Layout(layout.init_poses.copy(), layout.obst_xy.copy(), layout.goal.copy(), layout.slots)
        self.env.reset(self.layout.init_poses, self.layout.obst_xy)
        self.poses = self.env.poses()
        self.t[:] = 0; self.ret[:] = 0; self.form_sum[:] = 0

    def reset_where(self, mask):
        n = int(mask.sum())
        if not n:
            return
        init, obst, goal, _ = self.stream.take(n)
        full_init = self.layout.init_poses.copy(); full_init[mask] = init
        full_obst = self.layout.obst_xy.copy(); full_obst[mask] = obst
        self.layout.init_poses = full_init; self.layout.obst_xy = full_obst
        self.layout.goal = self.layout.goal.copy(); self.layout.goal[mask] = goal
        self.env.reset_subset(mask, full_init, full_obst)
        self.poses = self.env.poses()
        self.t[mask] = 0; self.ret[mask] = 0; self.form_sum[mask] = 0


class FormationTrainEnv:
    """One TrainVmasEnv per obstacle density, stepped together; auto-reset on episode end."""

    def __init__(self, cfg: StudyConfig, run_seed: int, envs_per_density: int, densities=None, vmas_seed=None):
        self.cfg = cfg
        self.pc = cfg.protocol_config()
        self.oc = cfg.obs_config()
        self.rc = cfg.reward_config()
        fc = cfg.layout_filter_config()
        self.lo = np.array([fc.v_min, -fc.w_max])
        self.hi = np.array([fc.v_max, fc.w_max])
        self.v_scale, self.w_scale = fc.v_abs_max, fc.w_max
        dens = cfg.densities if densities is None else tuple(densities)
        vs = run_seed if vmas_seed is None else vmas_seed
        self.groups = [_Group(cfg, run_seed, K, envs_per_density, vs) for K in dens]
        self.n_envs = envs_per_density * len(self.groups)

    def obs(self):
        return np.concatenate([build_obs(g.poses, g.layout, self.oc) for g in self.groups], axis=0)

    def reset(self, layouts=None):
        for i, g in enumerate(self.groups):
            g.reset(None if layouts is None else layouts[i])
            self._check_start(g)
        return self.obs()

    def _check_start(self, g):
        hit_o, hit_r, _ = score_motion(g.poses, np.zeros(g.poses.shape[:2] + (2,)), g.poses, g.layout.obst_xy, self.pc)
        if (hit_o | hit_r).any() or goal_reached(g.poses, g.layout, self.pc).any():
            raise RuntimeError("start state overlaps or is already at the goal; layouts must satisfy A5")

    def step(self, raw):
        raw = np.asarray(raw, dtype=np.float64)
        outs = []
        e0 = 0
        episodes = []
        for g in self.groups:
            u = np.clip(raw[e0:e0 + g.E], self.lo, self.hi)
            e0 += g.E
            p0 = g.poses
            g.env.step(u)
            p1 = g.env.poses()
            hit_o, hit_r, _ = score_motion(p0, u, p1, g.layout.obst_xy, self.pc)
            collided = hit_o | hit_r
            reached = ~collided & goal_reached(p1, g.layout, self.pc)
            g.t += 1
            truncated = ~collided & ~reached & (g.t >= self.pc.max_steps)
            terminated = collided | reached

            d0 = np.linalg.norm(p0[..., :2].mean(axis=1) - g.layout.goal, axis=1)
            d1 = np.linalg.norm(p1[..., :2].mean(axis=1) - g.layout.goal, axis=1)
            ferr = formation_error(p1, g.layout.slots)
            ctrl = ((u[..., 0] / self.v_scale) ** 2 + (u[..., 1] / self.w_scale) ** 2).mean(axis=1)
            rc = self.rc
            r = (rc.w_progress * (d0 - d1) - rc.w_formation * ferr - rc.w_control * ctrl
                 - rc.w_collision_obst * hit_o - rc.w_collision_rr * hit_r)
            if rc.w_proximity:
                r = r - rc.w_proximity * proximity_penalty(p1, g.layout.obst_xy, self.pc, rc.proximity_margin)
            g.ret += r
            g.form_sum += ferr
            g.poses = p1
            final_obs = build_obs(p1, g.layout, self.oc)

            done = terminated | truncated
            for b in np.flatnonzero(done):
                oc = Outcome.COLLISION if collided[b] else Outcome.GOAL if reached[b] else Outcome.TIMEOUT
                episodes.append(dict(density=g.K, outcome=int(oc), length=int(g.t[b]), ret=float(g.ret[b]),
                                     form_err_mean=float(g.form_sum[b] / g.t[b]),
                                     collision_kind=1 if hit_o[b] else 2 if hit_r[b] else 0))
            g.reset_where(done)
            outs.append((r, terminated, truncated, final_obs))
        obs = self.obs()
        r, term, trunc, fobs = (np.concatenate(x, axis=0) for x in zip(*outs))
        return StepOut(obs, r, term, trunc, fobs, episodes)
