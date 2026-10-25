"""
Layout solvability audit: geometry only, no policy and no filter.

For one sampled layout, is there a collision-free way to the goal within the episode horizon?
Three questions, each answered FOUND / BLOCKED / UNDETERMINED:

  (a) individual   each robot alone, from its start pose to its own goal  goal + slot_i,
                   obstacles inflated by the robot radius (+ margin);
  (b) team, free   the three robots reach their goals and never come closer than 2R (+ margin)
                   to one another; no formation constraint during transit (the protocol checks
                   formation only at the goal: formation_error <= formation_tol);
  (c) team, tol    as (b) but the formation deformation stays within the goal tolerance during
                   the whole transit: mean slot error  (1/N) sum_i |x_i - c - s_i| <= formation_tol.

Method
------
Configuration space = the plane minus discs (obstacle centre, radius R + R_o + margin), on a
square grid of cell size h over a box whose outer ring is free (no arena walls, A6: any route that
leaves the box can be re-routed along the ring, so restricting to the box loses nothing).

* FOUND is constructive and exact: a shortest path on the CERTAINLY-FREE cells (cell entirely
  outside every disc, 8-connected) is shortcut, and every segment of the final polyline is checked
  exactly against every disc. The unicycle follows the polyline by turning in place and driving
  straight (v <= v_max, |w| <= w_max, zero-order hold at dt): steps = sum ceil(L / (v_max dt)) +
  sum ceil(|turn| / (w_max dt)); FOUND requires steps <= max_steps.
    (b) simultaneously when the three verified polylines are pairwise at least 2R + margin apart;
        else the rigid-formation plan of (c) when it exists (slot spacing > 2R); else one at a time
        (each order tried), the others held as static discs of radius 2R + margin at their start
        or goal. Sequential steps add up.
    (c) robots first move one at a time onto their exact slots around the start centroid (the
        formation error along straight moves is convex in time, so checking the end points
        suffices), then the rigid triangle translates along a centroid polyline planned among
        discs centred at o_k - s_i (zero deformation).
* BLOCKED is a proof of impossibility at any horizon: start and goal lie in different
  8-connected components of the POSSIBLY-FREE cells (cell not inside any single disc), which
  over-approximate the free space (any continuous route visits an 8-connected chain of cells that
  each contain a free point).
    (b) blocked when some robot is blocked in (a).
    (c) relaxation of the deformation limit: robot i at c + s_i + e_i must be outside every disc,
        so |e_i| >= max_k (r - |c + s_i - o_k|)_+ and sum_i |e_i| <= N * formation_tol gives
        f(c) = sum_i max_k (r - |c + s_i - o_k|)_+ <= N * tol (f is N-Lipschitz, applied per cell).
* UNDETERMINED: neither (search failed without a proof, or the route found exceeds the horizon).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

FOUND, BLOCKED, UNDETERMINED = "found", "blocked", "undetermined"
EIGHT = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class AuditConfig:
    robot_r: float
    obst_r: float
    v_max: float
    w_max: float
    dt: float
    max_steps: int
    formation_tol: float
    h: float = 0.02                     # grid cell [m]
    margin: float = 0.002               # extra clearance on every disc [m] (scoring flags touching)
    box: tuple = (-2.6, 2.6, -2.0, 2.0)

    @classmethod
    def from_study(cls, cfg, **kw):
        pc = cfg.protocol_config()
        return cls(robot_r=pc.robot_r, obst_r=pc.obst_r, v_max=pc.v_max, w_max=pc.w_max, dt=pc.dt,
                   max_steps=pc.max_steps, formation_tol=pc.formation_tol, **kw)


class Grid:
    def __init__(self, ac: AuditConfig):
        x0, x1, y0, y1 = ac.box
        self.h = ac.h
        self.nx, self.ny = int(round((x1 - x0) / ac.h)), int(round((y1 - y0) / ac.h))
        self.x0, self.y0 = x0, y0
        xs = x0 + (np.arange(self.nx) + 0.5) * ac.h
        ys = y0 + (np.arange(self.ny) + 0.5) * ac.h
        self.X, self.Y = np.meshgrid(xs, ys)
        self.hd = ac.h * math.sqrt(2) / 2

    def cell(self, p):
        i = int(np.floor((p[1] - self.y0) / self.h))
        j = int(np.floor((p[0] - self.x0) / self.h))
        if not (0 <= i < self.ny and 0 <= j < self.nx):
            raise ValueError(f"point {p} outside the audit box")
        return i, j

    def center(self, i, j):
        return np.array([self.x0 + (j + 0.5) * self.h, self.y0 + (i + 0.5) * self.h])

    def masks(self, C, r):
        """certainly-free and possibly-free cell masks for discs (C [M,2], r [M])."""
        cons = np.ones((self.ny, self.nx), dtype=bool)
        opt = np.ones((self.ny, self.nx), dtype=bool)
        for (cx, cy), rr in zip(C, r):
            d = np.hypot(self.X - cx, self.Y - cy)
            cons &= d - self.hd >= rr
            opt &= ~(d + self.hd <= rr)
        return cons, opt

    def ring_is_free(self, cons):
        return cons[0].all() and cons[-1].all() and cons[:, 0].all() and cons[:, -1].all()

    def connected(self, mask, a, b):
        lab, _ = ndimage.label(mask, structure=EIGHT)
        la, lb = lab[self.cell(a)], lab[self.cell(b)]
        return la != 0 and la == lb

    def path(self, free, a, b):
        """Cell-centre polyline from a to b on 8-connected free cells, or None."""
        ia, ib = self.cell(a), self.cell(b)
        if not (free[ia] and free[ib]) or not self.connected(free, a, b):
            return None
        ny, nx = free.shape
        idx = np.arange(ny * nx).reshape(ny, nx)
        rows, cols, w = [], [], []
        for dy, dx, wt in ((0, 1, 1.0), (1, 0, 1.0), (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2))):
            sy = slice(0, ny - dy); ty = slice(dy, ny)
            sx, tx = (slice(0, nx - dx), slice(dx, nx)) if dx >= 0 else (slice(-dx, nx), slice(0, nx + dx))
            both = free[sy, sx] & free[ty, tx]
            rows.append(idx[sy, sx][both]); cols.append(idx[ty, tx][both]); w.append(np.full(both.sum(), wt))
        g = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(ny * nx,) * 2).tocsr()
        s, t = idx[ia], idx[ib]
        _, pred = dijkstra(g, directed=False, indices=s, return_predecessors=True)
        if pred[t] < 0 and s != t:
            return None
        chain = [t]
        while chain[-1] != s:
            chain.append(pred[chain[-1]])
        cells = [divmod(int(c), nx) for c in reversed(chain)]
        return np.array([a] + [self.center(i, j) for i, j in cells] + [b], dtype=np.float64)


# ------------------------------------------------------------------------------------------------
# exact geometry
# ------------------------------------------------------------------------------------------------
def seg_point_dist(a, b, P):
    """distance from segment a-b to points P [M,2]."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 == 0.0:
        return np.linalg.norm(P - a, axis=1)
    t = np.clip(((P - a) @ ab) / L2, 0.0, 1.0)
    return np.linalg.norm(a + t[:, None] * ab - P, axis=1)


def seg_clear(a, b, C, r):
    return bool((seg_point_dist(a, b, C) >= r).all()) if len(C) else True


def seg_seg_dist(a, b, c, d):
    """minimum distance between segments a-b and c-d (exact)."""
    def orient(p, q, s):
        return (q[0] - p[0]) * (s[1] - p[1]) - (q[1] - p[1]) * (s[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    if (o1 * o2 < 0) and (o3 * o4 < 0):
        return 0.0
    return float(min(seg_point_dist(a, b, np.array([c, d])).min(), seg_point_dist(c, d, np.array([a, b])).min()))


def polyline_dist(P, Q):
    return min(seg_seg_dist(P[i], P[i + 1], Q[j], Q[j + 1]) for i in range(len(P) - 1) for j in range(len(Q) - 1))


def shortcut(P, C, r):
    """Greedy shortcut; every returned segment is exactly clear of the discs, else None."""
    out = [P[0]]
    i = 0
    n = len(P)
    while i < n - 1:
        j = n - 1
        while j > i and not seg_clear(P[i], P[j], C, r):
            j -= 1
        if j == i:
            return None
        out.append(P[j])
        i = j
    return np.array(out)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def polyline_steps(P, heading0, ac: AuditConfig):
    """ZOH steps to follow polyline P by turn-in-place and straight driving, starting at heading0."""
    steps, th = 0, heading0
    for a, b in zip(P[:-1], P[1:]):
        L = float(np.linalg.norm(b - a))
        if L < 1e-12:
            continue
        psi = math.atan2(b[1] - a[1], b[0] - a[0])
        steps += math.ceil(abs(wrap(psi - th)) / (ac.w_max * ac.dt) - 1e-9)
        steps += math.ceil(L / (ac.v_max * ac.dt) - 1e-9)
        th = psi
    return steps


def path_length(P):
    return float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum())


# ------------------------------------------------------------------------------------------------
# audit of one layout
# ------------------------------------------------------------------------------------------------
def _plan(grid, C, r, a, b):
    cons, _ = grid.masks(C, r)
    P = grid.path(cons, a, b)
    return None if P is None else shortcut(P, C, r)


def audit_layout(init_poses, obst_xy, goal, slots, ac: AuditConfig, grid: Grid | None = None, return_paths=False):
    grid = grid or Grid(ac)
    N = len(init_poses)
    starts = np.asarray(init_poses, dtype=np.float64)[:, :2]
    heads = np.asarray(init_poses, dtype=np.float64)[:, 2]
    goals = np.asarray(goal, dtype=np.float64)[None] + np.asarray(slots, dtype=np.float64)
    O = np.asarray(obst_xy, dtype=np.float64).reshape(-1, 2)
    r_o = ac.robot_r + ac.obst_r + ac.margin
    r_rr = 2 * ac.robot_r + ac.margin
    cons_o, opt_o = grid.masks(O, np.full(len(O), r_o))
    if not grid.ring_is_free(cons_o):
        raise ValueError("audit box outer ring is not free; enlarge the box")
    out = {}

    # (a) individual -----------------------------------------------------------------------
    ind_paths, ind_steps, ind_status = [], [], []
    for i in range(N):
        P = grid.path(cons_o, starts[i], goals[i])
        P = None if P is None else shortcut(P, O, np.full(len(O), r_o))
        if P is not None:
            st = polyline_steps(P, heads[i], ac)
            ind_paths.append(P); ind_steps.append(st)
            ind_status.append(FOUND if st <= ac.max_steps else UNDETERMINED)
        else:
            ind_paths.append(None); ind_steps.append(None)
            ind_status.append(UNDETERMINED if grid.connected(opt_o, starts[i], goals[i]) else BLOCKED)
    out["a_status"] = (FOUND if all(s == FOUND for s in ind_status) else
                       BLOCKED if any(s == BLOCKED for s in ind_status) else UNDETERMINED)
    out["a_robots_found"] = sum(s == FOUND for s in ind_status)
    out["a_steps_max"] = max(ind_steps) if all(s is not None for s in ind_steps) else None
    out["a_len_max"] = max(path_length(P) for P in ind_paths) if all(P is not None for P in ind_paths) else None

    # (c) team, deformation <= formation_tol throughout ----------------------------------------
    c_status, c_steps = UNDETERMINED, None
    c0 = starts.mean(axis=0)
    targets0 = c0[None] + slots
    # phase 0: each robot onto its exact slot, one at a time
    pos = starts.copy()
    steps0, ok0 = 0, True
    for i in range(N):
        e0 = np.linalg.norm((pos - pos.mean(0)) - slots, axis=1).mean()
        nxt = pos.copy(); nxt[i] = targets0[i]
        e1 = np.linalg.norm((nxt - nxt.mean(0)) - slots, axis=1).mean()
        others = np.delete(pos, i, axis=0)
        if not (seg_clear(pos[i], targets0[i], O, np.full(len(O), r_o)) and
                seg_clear(pos[i], targets0[i], others, np.full(len(others), r_rr)) and
                max(e0, e1) <= ac.formation_tol):
            ok0 = False
            break
        steps0 += polyline_steps(np.array([pos[i], targets0[i]]), heads[i], ac)
        pos = nxt
    heads_after = heads.copy()
    for i in range(N):
        d = targets0[i] - starts[i]
        if np.linalg.norm(d) > 1e-12:
            heads_after[i] = math.atan2(d[1], d[0])
    if ok0:
        Cf = (O[None, :, :] - slots[:, None, :]).reshape(-1, 2)
        rf = np.full(len(Cf), r_o)
        P = _plan(grid, Cf, rf, c0, goal)
        if P is not None:
            # every robot turns in place to each segment direction, then all translate together
            st = steps0 + max(polyline_steps(P, h0, ac) for h0 in heads_after)
            if st <= ac.max_steps:
                c_status, c_steps = FOUND, st
    if c_status != FOUND:
        if out["a_status"] == BLOCKED:
            c_status = BLOCKED
        else:
            f = np.zeros_like(grid.X)
            r = ac.robot_r + ac.obst_r + ac.margin
            for i in range(N):
                depth = np.zeros_like(grid.X)
                for o in O:
                    depth = np.maximum(depth, r - np.hypot(grid.X + slots[i, 0] - o[0], grid.Y + slots[i, 1] - o[1]))
                f += depth
            possible = f - N * grid.hd <= N * ac.formation_tol
            if not grid.connected(possible, c0, goal):
                c_status = BLOCKED
    out["c_status"], out["c_steps"] = c_status, c_steps
    if return_paths:
        out["a_paths"] = ind_paths
        out["c_path"] = P if c_status == FOUND else None
    # (b) team, no formation constraint in transit -------------------------------------------
    b_status, b_steps, b_mode = UNDETERMINED, None, ""
    if out["a_status"] == BLOCKED:
        b_status = BLOCKED
    else:
        if all(P is not None for P in ind_paths) and all(
                polyline_dist(ind_paths[i], ind_paths[j]) >= r_rr for i, j in itertools.combinations(range(N), 2)):
            st = max(ind_steps)
            if st <= ac.max_steps:
                b_status, b_steps, b_mode = FOUND, st, "simultaneous"
        if b_status != FOUND and c_status == FOUND:
            # the rigid-formation plan keeps every pair at the slot spacing (> 2R): it is a team plan too
            b_status, b_steps, b_mode = FOUND, c_steps, "formation"
        if b_status != FOUND:
            best = None
            for order in itertools.permutations(range(N)):
                total, ok = 0, True
                for k, i in enumerate(order):
                    others = np.array([goals[j] if j in order[:k] else starts[j] for j in range(N) if j != i])
                    C = np.concatenate([O, others]); r = np.r_[np.full(len(O), r_o), np.full(len(others), r_rr)]
                    P = _plan(grid, C, r, starts[i], goals[i])
                    if P is None:
                        ok = False
                        break
                    total += polyline_steps(P, heads[i], ac)
                if ok and (best is None or total < best[0]):
                    best = (total, order)
            if best is not None and best[0] <= ac.max_steps:
                b_status, b_steps, b_mode = FOUND, best[0], "sequential " + "".join(map(str, best[1]))
    out["b_status"], out["b_steps"], out["b_mode"] = b_status, b_steps, b_mode

    return out
