"""Solvability audit on constructed layouts: free space, an enclosed start, a gap one robot fits
through but the triangle does not, and the horizon rule."""
from dataclasses import replace

import numpy as np
from scipy.optimize import brentq

from cbf_study.config import StudyConfig
from cbf_study.scenario import triangle_slots
from cbf_study.solvability import (BLOCKED, FOUND, UNDETERMINED, AuditConfig, audit_layout, seg_clear,
                                   seg_point_dist, seg_seg_dist)

CFG = StudyConfig.load()
AC = AuditConfig.from_study(CFG)
SLOTS = triangle_slots(3, 0.30)
C0 = np.array([-1.2, 0.0])
GOAL = np.array([1.5, 0.0])


def start_poses(c0=C0):
    xy = c0 + SLOTS
    return np.concatenate([xy, np.zeros((3, 1))], axis=1)


def ring(center, gap_open, gap_reg=0.10, n_reg=11, R_o=0.20, open_angle=0.0):
    """obstacles on a circle: n_reg gaps of gap_reg between surfaces and one gap of gap_open."""
    d1, d2 = 2 * R_o + gap_reg, 2 * R_o + gap_open
    f = lambda rho: n_reg * 2 * np.arcsin(d1 / 2 / rho) + 2 * np.arcsin(d2 / 2 / rho) - 2 * np.pi
    rho = brentq(f, 0.5, 5.0)
    a1, a2 = 2 * np.arcsin(d1 / 2 / rho), 2 * np.arcsin(d2 / 2 / rho)
    angs = open_angle + a2 / 2 + a1 * np.arange(n_reg + 1)
    return center + rho * np.stack([np.cos(angs), np.sin(angs)], axis=1)


def test_free_space_is_found_with_expected_steps():
    out = audit_layout(start_poses(), np.zeros((0, 2)), GOAL, SLOTS, AC)
    assert out["a_status"] == out["b_status"] == out["c_status"] == FOUND
    # straight 2.7 m at 0.05 m per step = 54 steps, headings already along +x
    assert out["a_steps_max"] == 54 and out["c_steps"] == 54


def test_enclosed_start_is_proven_blocked():
    obst = ring(C0, gap_open=0.10)                       # every gap 0.1 m < robot diameter 0.2 m
    out = audit_layout(start_poses(), obst, GOAL, SLOTS, AC)
    assert out["a_status"] == out["b_status"] == out["c_status"] == BLOCKED


def test_single_file_gap_passable_by_robots_not_by_the_triangle():
    obst = ring(C0, gap_open=0.26)                       # one gap a single robot (0.2 + margins) fits through
    out = audit_layout(start_poses(), obst, GOAL, SLOTS, AC)
    assert out["a_status"] == FOUND and out["b_status"] == FOUND
    assert out["b_mode"].startswith("sequential")        # they must take turns through the gap
    # the rigid triangle cannot pass; the deformation relaxation is too weak to PROVE it blocked here,
    # and the audit must not claim more than it can show
    assert out["c_status"] == UNDETERMINED
    # a gap narrower than the grid can resolve is undetermined for single robots, never 'blocked'
    out = audit_layout(start_poses(), ring(C0, gap_open=0.24), GOAL, SLOTS, AC)
    assert out["a_status"] == UNDETERMINED


def test_route_beyond_horizon_is_undetermined_not_blocked():
    out = audit_layout(start_poses(), np.zeros((0, 2)), GOAL, SLOTS, replace(AC, max_steps=30))
    assert out["a_status"] == out["b_status"] == out["c_status"] == UNDETERMINED


def test_exact_segment_geometry():
    rng = np.random.default_rng(0)
    for _ in range(300):
        a, b, c, d = rng.uniform(-1, 1, (4, 2))
        t = np.linspace(0, 1, 801)[:, None]
        P, Q = a + t * (b - a), c + t * (d - c)
        tol = 2 * np.sqrt(8) / 800                       # sampling error bound of the brute-force reference
        brute = np.min(np.linalg.norm(P[:, None] - Q[None], axis=2))
        exact = seg_seg_dist(a, b, c, d)
        assert exact <= brute + 1e-9 and brute - exact < tol
        o = rng.uniform(-1, 1, (1, 2))
        dmin = np.linalg.norm(P - o, axis=1).min()
        assert -1e-9 <= dmin - seg_point_dist(a, b, o)[0] < tol
        r = rng.uniform(0.05, 0.5)
        if abs(dmin - r) > tol:
            assert seg_clear(a, b, o, np.array([r])) == (dmin >= r)
