"""
Layout sampling and a STAND-IN policy.

Everything numeric here is a placeholder pending advisor approval (formation, densities, limits).
The stand-in policy is a test double: a deterministic, obstacle-blind go-to-slot controller. It
exists so the protocol and the filter can be exercised end to end before the MAPPO policy is
trained. It is NOT the study policy and no result obtained with it belongs in the study.
The harness accepts any callable policy(poses [B,N,3], layout) -> actions [B,N,2].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cbf_qp import FilterConfig, barrier_point


@dataclass
class Layout:
    init_poses: np.ndarray    # [B,N,3]
    obst_xy: np.ndarray       # [B,K,2]
    goal: np.ndarray          # [B,2] goal for the team centroid
    slots: np.ndarray         # [N,2] formation slots relative to the centroid


@dataclass(frozen=True)
class ScenarioConfig:
    n_agents: int = 3
    slot_radius: float = 0.30
    start_centroid: tuple = (-1.5, 0.0)
    goal_centroid: tuple = (1.5, 0.0)
    obst_x: tuple = (-0.8, 0.8)
    obst_y: tuple = (-0.9, 0.9)
    obst_gap: float = 0.10          # min free gap between obstacle surfaces
    edge_clear: float = 0.10        # obstacle-centre clearance from start and goal centroids
    start_xy_std: float = 0.03      # start position noise around each slot
    start_th_std: float = 0.15      # start heading noise


def triangle_slots(n, radius):
    ang = np.pi / 2 + 2 * np.pi * np.arange(n) / n
    return radius * np.stack([np.cos(ang), np.sin(ang)], axis=1)


def sample_layout(seed, batch, n_obst, fcfg: FilterConfig, sc: ScenarioConfig = ScenarioConfig()):
    """Seeded sampler. The same seed always returns the same layout, which is how filter-on and
    filter-off trials get identical initial states. Initial states satisfy h >= 0 for every
    barrier (premise of the safety lemma) and the goal region is free of obstacles."""
    rng = np.random.default_rng(seed)
    slots = triangle_slots(sc.n_agents, sc.slot_radius)
    start = np.array(sc.start_centroid)
    goal = np.array(sc.goal_centroid)
    init = np.zeros((batch, sc.n_agents, 3))
    obst = np.zeros((batch, n_obst, 2))
    for b in range(batch):
        while True:   # start poses
            xy = start + slots + rng.normal(0, sc.start_xy_std, slots.shape)
            th = rng.normal(0, sc.start_th_std, sc.n_agents)
            pose = np.concatenate([xy, th[:, None]], axis=1)
            p = barrier_point(pose, fcfg.ell)
            d = np.linalg.norm(p[:, None] - p[None], axis=2) + np.eye(sc.n_agents) * 1e9
            if (d >= fcfg.d_rr + 0.02).all():
                init[b] = pose
                break
        k = 0
        tries = 0
        restarts = 0
        while k < n_obst:
            tries += 1
            if tries > 10000:
                # Sampler v1.1: the obstacles placed so far can leave no room for the rest (seen at
                # K = 8, e.g. layout seed 1108377). Resample this layout's obstacles from scratch
                # instead of failing. The random stream is consumed exactly as before up to this
                # point, so every layout that v1 could produce is unchanged.
                restarts += 1
                if restarts > 100:
                    raise RuntimeError("could not place obstacles; relax ScenarioConfig")
                k, tries = 0, 0
                continue
            c = np.array([rng.uniform(*sc.obst_x), rng.uniform(*sc.obst_y)])
            if np.linalg.norm(c - start) < sc.edge_clear + fcfg.d_obst + sc.slot_radius:
                continue
            if np.linalg.norm(c - goal) < sc.edge_clear + fcfg.d_obst + sc.slot_radius:
                continue
            if k and (np.linalg.norm(obst[b, :k] - c, axis=1) < 2 * fcfg.obst_r + sc.obst_gap).any():
                continue
            obst[b, k] = c
            k += 1
    return Layout(init, obst, np.tile(goal, (batch, 1)), slots)


class SlotPolicy:
    """Deterministic go-to-slot controller (test double, obstacle-blind)."""

    def __init__(self, v_max, w_max, k_v=1.0, k_w=2.0):
        self.v_max, self.w_max, self.k_v, self.k_w = v_max, w_max, k_v, k_w

    def __call__(self, poses, layout):
        target = layout.goal[:, None, :] + layout.slots[None]
        e = target - poses[..., :2]
        dist = np.linalg.norm(e, axis=2)
        err = np.arctan2(e[..., 1], e[..., 0]) - poses[..., 2]
        err = (err + np.pi) % (2 * np.pi) - np.pi
        v = np.clip(self.k_v * dist * np.cos(err), 0.0, self.v_max)
        w = np.clip(self.k_w * err, -self.w_max, self.w_max)
        return np.stack([v, w], axis=2)
