"""
Exact unicycle motion and between-step (swept-path) collision scoring.

All functions are independent of VMAS. Poses are (x, y, theta) of the AXLE CENTRE, which is also
the centre of the circular body footprint of radius R.

Why this module exists
----------------------
Checking collisions only at control-step endpoints can miss a robot that crosses an obstacle or
another robot between samples. With drag = 0 and RK4 DiffDrive, VMAS executes the commanded
(v, w) as an exact arc (gate test), so the continuous path inside one control step is known in
closed form and can be checked.

Conservative substep rule (robot-obstacle and robot-robot)
----------------------------------------------------------
Sample the arc(s) at m + 1 equally spaced times, tau = dt / m. The centre-to-centre distance
d(t) is Lipschitz with constant L = |v_i| (static obstacle) or |v_i| + |v_j| (two robots), so

    true min separation >= sampled min separation - L * tau / 2.

A step is scored as a collision when that lower bound is < 0. This never misses a true overlap;
it can flag a near miss whose true separation is below L * tau / 2 (< 1 mm for the default
settings), and it uses the same rule for filter-on and filter-off trials.
"""
from __future__ import annotations

import numpy as np

DEFAULT_SUBSTEPS = 64


def arc_poses(pose, v, w, dt, m=DEFAULT_SUBSTEPS):
    """Exact poses along a constant-(v, w) arc at times k*dt/m, k=0..m.

    pose: [N,3], v: [N], w: [N]  ->  [N, m+1, 3]
    """
    pose = np.asarray(pose, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    t = np.linspace(0.0, dt, m + 1)[None, :]                      # [1, m+1]
    th0 = pose[:, 2:3]
    wt = w[:, None] * t
    small = np.abs(w[:, None]) < 1e-9
    safe_w = np.where(small, 1.0, w[:, None])
    x = np.where(small, pose[:, 0:1] + v[:, None] * t * np.cos(th0),
                 pose[:, 0:1] + (v[:, None] / safe_w) * (np.sin(th0 + wt) - np.sin(th0)))
    y = np.where(small, pose[:, 1:2] + v[:, None] * t * np.sin(th0),
                 pose[:, 1:2] + (v[:, None] / safe_w) * (-np.cos(th0 + wt) + np.cos(th0)))
    return np.stack([x, y, th0 + wt], axis=2)


def endpoint_sep_obstacle(pose0, v, w, dt, centre, r_sum):
    """Endpoint-only separation to a static circle (what a naive per-step check would see)."""
    end = arc_poses(pose0, v, w, dt, 1)[:, -1, :2]
    start = np.asarray(pose0)[:, :2]
    return np.minimum(np.linalg.norm(start - centre, axis=1),
                      np.linalg.norm(end - centre, axis=1)) - r_sum


def swept_sep_lb_obstacle(pose0, v, w, dt, centre, r_sum, m=DEFAULT_SUBSTEPS):
    """Conservative lower bound on the minimum separation to a static circle during one step.

    centre: [N,2] or [2]; r_sum: scalar or [N]. Returns [N].
    """
    pts = arc_poses(pose0, v, w, dt, m)[:, :, :2]                  # [N,m+1,2]
    c = np.broadcast_to(np.asarray(centre, dtype=np.float64), (pts.shape[0], 2))[:, None, :]
    d = np.linalg.norm(pts - c, axis=2).min(axis=1)
    L = np.abs(np.asarray(v, dtype=np.float64))
    return d - r_sum - L * (dt / m) / 2.0


def endpoint_sep_robots(pose_i, v_i, w_i, pose_j, v_j, w_j, dt, r_sum):
    ei = arc_poses(pose_i, v_i, w_i, dt, 1)[:, :, :2]
    ej = arc_poses(pose_j, v_j, w_j, dt, 1)[:, :, :2]
    return np.linalg.norm(ei - ej, axis=2).min(axis=1) - r_sum


def swept_sep_lb_robots(pose_i, v_i, w_i, pose_j, v_j, w_j, dt, r_sum, m=DEFAULT_SUBSTEPS):
    """Conservative lower bound on min separation of two moving robots during one step."""
    pi = arc_poses(pose_i, v_i, w_i, dt, m)[:, :, :2]
    pj = arc_poses(pose_j, v_j, w_j, dt, m)[:, :, :2]
    d = np.linalg.norm(pi - pj, axis=2).min(axis=1)
    L = np.abs(np.asarray(v_i, dtype=np.float64)) + np.abs(np.asarray(v_j, dtype=np.float64))
    return d - r_sum - L * (dt / m) / 2.0


def exact_arc_point_distance(pose, v, w, dt, point):
    """EXACT minimum distance from the centre path of one step to a point (test oracle, scalars).

    Path: constant (v, w) over dt. Straight segment if w ~ 0, else a circular arc of signed radius
    R = v / w about C = (x0 - R sin th0, y0 + R cos th0), with position C + R (sin psi, -cos psi),
    psi = th0 + w t.
    """
    x0, y0, th0 = (float(a) for a in pose)
    P = np.array([float(point[0]), float(point[1])])
    end = arc_poses(np.array([[x0, y0, th0]]), np.array([v]), np.array([w]), dt, 1)[0, -1, :2]
    ends = min(np.linalg.norm(np.array([x0, y0]) - P), np.linalg.norm(end - P))
    if abs(w) < 1e-9:
        a = np.array([x0, y0]); ab = end - a
        denom = float(ab @ ab)
        if denom == 0.0:
            return float(ends)
        s = float(np.clip(((P - a) @ ab) / denom, 0.0, 1.0))
        return float(np.linalg.norm(a + s * ab - P))
    R = v / w
    if R == 0.0:
        return float(ends)
    C = np.array([x0 - R * np.sin(th0), y0 + R * np.cos(th0)])
    off = P - C
    dist_c = float(np.linalg.norm(off))
    if dist_c == 0.0:
        return float(abs(R))
    u = off / dist_c
    s = np.sign(R)
    psi = np.arctan2(s * u[0], -s * u[1])                  # heading at which the arc is closest to P
    delta = ((psi - th0) * np.sign(w)) % (2 * np.pi)
    if delta <= abs(w) * dt:
        return float(abs(dist_c - abs(R)))
    return float(ends)
