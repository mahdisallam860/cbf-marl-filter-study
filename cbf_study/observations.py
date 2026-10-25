"""
Policy observations computed from (poses, layout) only.

The same function is called by the training environment (train_env.py) and by the frozen policy
at evaluation (policies.py), so the policy cannot see anything at evaluation that it did not see
in training, and nothing the filter computes ever reaches it.

Per robot i, everything in robot i's body frame (rotation by -theta_i):
    slot error      (centroid + slot_i) - x_i                                     2
    goal            goal - centroid, norm clipped at goal_clip                    2
    teammates       for j != i in index order: x_j - x_i, cos/sin(theta_j - theta_i)   4 (N-1)
    obstacles       k_obst nearest by centre distance: c - x_i, surface gap
                    |c - x_i| - (R + R_o), present flag; absent or beyond
                    obs_range -> (0, 0, obs_range, 0)                             4 k_obst
    agent id        one-hot                                                       N
"""
from __future__ import annotations

import numpy as np

from .config import ObsConfig


def obs_dim(oc: ObsConfig) -> int:
    return 4 + 4 * (oc.n_agents - 1) + 4 * oc.k_obst + oc.n_agents


def _to_body(vec, c, s):
    return np.stack([c * vec[..., 0] + s * vec[..., 1], -s * vec[..., 0] + c * vec[..., 1]], axis=-1)


def build_obs(poses, layout, oc: ObsConfig) -> np.ndarray:
    """poses [B,N,3], layout with obst_xy [B,K,2], goal [B,2], slots [N,2] -> float32 [B,N,obs_dim]."""
    poses = np.asarray(poses, dtype=np.float64)
    B, N, _ = poses.shape
    if N != oc.n_agents:
        raise ValueError(f"expected {oc.n_agents} robots, got {N}")
    xy, th = poses[..., :2], poses[..., 2]
    c, s = np.cos(th), np.sin(th)
    cen = xy.mean(axis=1)                                                   # [B,2]
    slots = np.asarray(layout.slots, dtype=np.float64)

    slot_err = _to_body(cen[:, None] + slots[None] - xy, c, s)             # [B,N,2]
    g = np.asarray(layout.goal, dtype=np.float64) - cen
    gn = np.linalg.norm(g, axis=1, keepdims=True)
    g = g * np.minimum(1.0, oc.goal_clip / np.maximum(gn, 1e-12))
    goal = _to_body(np.broadcast_to(g[:, None], (B, N, 2)), c, s)          # [B,N,2]

    others = []
    for i in range(N):
        js = [j for j in range(N) if j != i]
        rel = xy[:, js] - xy[:, i:i + 1]                                    # [B,N-1,2]
        rel = _to_body(rel, c[:, i:i + 1], s[:, i:i + 1])
        dth = th[:, js] - th[:, i:i + 1]
        others.append(np.concatenate([rel, np.cos(dth)[..., None], np.sin(dth)[..., None]], axis=-1).reshape(B, -1))
    mates = np.stack(others, axis=1)                                         # [B,N,4(N-1)]

    obst = np.asarray(layout.obst_xy, dtype=np.float64)
    K = obst.shape[1]
    ko = oc.k_obst
    feat = np.zeros((B, N, ko, 4))
    feat[..., 2] = oc.obs_range
    if K:
        rel = obst[:, None, :, :] - xy[:, :, None, :]                       # [B,N,K,2]
        d = np.linalg.norm(rel, axis=3)
        order = np.argsort(d, axis=2, kind="stable")[:, :, :ko]
        take = min(ko, K)
        rel_k = np.take_along_axis(rel, order[..., None], axis=2)[:, :, :take]
        d_k = np.take_along_axis(d, order, axis=2)[:, :, :take]
        present = d_k <= oc.obs_range
        body = _to_body(rel_k, c[..., None], s[..., None])
        gap = d_k - (oc.robot_r + oc.obst_r)
        feat[:, :, :take, 0:2] = np.where(present[..., None], body, 0.0)
        feat[:, :, :take, 2] = np.where(present, gap, oc.obs_range)
        feat[:, :, :take, 3] = present
    eye = np.broadcast_to(np.eye(N), (B, N, N))
    out = np.concatenate([slot_err, goal, mates, feat.reshape(B, N, -1), eye], axis=2)
    return out.astype(np.float32)
