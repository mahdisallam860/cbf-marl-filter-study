"""Barrier geometry, relative degree, footprint cover, margin bound and the one-period lemma."""
import numpy as np
import pytest

from cbf_study.cbf_qp import (CBFQPFilter, FilterConfig, barrier_point, barrier_values,
                              margin_obstacle, margin_robot, rot_jacobian)
from cbf_study.geometry import arc_poses
from conftest import make_cfgs


def test_pdot_equals_J_u_and_relative_degree_one():
    fc, _ = make_cfgs()
    rng = np.random.default_rng(0)
    for _ in range(50):
        pose = np.array([[*rng.uniform(-1, 1, 2), rng.uniform(-np.pi, np.pi)]])
        v, w = rng.uniform(-0.5, 0.5), rng.uniform(-1, 1)
        eps = 1e-6
        p0 = barrier_point(arc_poses(pose, [v], [w], eps, 1)[:, 0], fc.ell)[0]
        p1 = barrier_point(arc_poses(pose, [v], [w], eps, 1)[:, 1], fc.ell)[0]
        J = rot_jacobian(pose[:, 2], fc.ell)[0]
        assert np.allclose((p1 - p0) / eps, J @ np.array([v, w]), atol=1e-5)
        assert abs(np.linalg.det(J) - fc.ell) < 1e-12        # invertible for l > 0: h_dot depends on u


def test_ell_zero_is_rejected():
    with pytest.raises(ValueError):
        make_cfgs(ell=0.0)[0].validate()
    # at l = 0 the control enters pdot only through v: J has rank 1
    J0 = rot_jacobian(np.array([0.7]), 0.0)[0]
    assert np.linalg.matrix_rank(J0) == 1


def test_footprint_cover_robot_obstacle_and_robot_robot():
    fc, _ = make_cfgs()
    rng = np.random.default_rng(1)
    n_o = n_r = 0
    for _ in range(20000):
        pose = np.array([[rng.uniform(-1, 1, 2).tolist() + [rng.uniform(-np.pi, np.pi)],
                          rng.uniform(-1, 1, 2).tolist() + [rng.uniform(-np.pi, np.pi)]]]).reshape(1, 2, 3)
        obst = rng.uniform(-1, 1, (1, 1, 2))
        h_o, h_r = barrier_values(fc, pose, obst)
        if (h_o >= 0).all():                                 # barrier says safe ...
            n_o += 1
            for i in range(2):                               # ... so every body point clears the obstacle
                ang = rng.uniform(0, 2 * np.pi, 200)
                rad = fc.robot_r * np.sqrt(rng.uniform(0, 1, 200))
                pts = pose[0, i, :2] + np.stack([rad * np.cos(ang), rad * np.sin(ang)], 1)
                assert np.linalg.norm(pts - obst[0, 0], axis=1).min() >= fc.obst_r - 1e-12
        if h_r[0, 0, 1] >= 0:
            n_r += 1
            assert np.linalg.norm(pose[0, 0, :2] - pose[0, 1, :2]) >= 2 * fc.robot_r - 1e-12
    assert n_o > 1000 and n_r > 1000


def test_hddot_bounds_hold_numerically():
    """|hddot| <= M used in the margin, checked on exact arcs by second differences."""
    fc, _ = make_cfgs()
    rng = np.random.default_rng(2)
    e, m = 1e-4, 400
    worst_o = worst_r = 0.0
    for _ in range(300):
        poses = np.array([[[*rng.uniform(-1, 1, 2), rng.uniform(-np.pi, np.pi)] for _ in range(2)]])
        u = np.stack([rng.uniform(fc.v_min, fc.v_max, 2), rng.uniform(-fc.w_max, fc.w_max, 2)], 1)
        c = rng.uniform(-1, 1, 2)
        traj = [arc_poses(poses[:, i], u[None, i, 0], u[None, i, 1], fc.dt, m)[0] for i in range(2)]
        p = [barrier_point(t, fc.ell) for t in traj]               # [m+1,2]
        t = np.linspace(0, fc.dt, m + 1)
        step = t[1] - t[0]
        h_o = np.linalg.norm(p[0] - c, axis=1) ** 2
        h_r = np.linalg.norm(p[0] - p[1], axis=1) ** 2
        for h, M0, kind in ((h_o, None, "o"), (h_r, None, "r")):
            dd = (h[2:] - 2 * h[1:-1] + h[:-2]) / step ** 2
            d0 = np.linalg.norm(p[0][0] - c) if kind == "o" else np.linalg.norm(p[0][0] - p[1][0])
            M = 2 * margin_obstacle(fc, d0) / fc.dt if kind == "o" else 2 * margin_robot(fc, d0) / fc.dt
            assert np.abs(dd).max() <= M * (1 + 1e-6)
            if kind == "o":
                worst_o = max(worst_o, np.abs(dd).max() / M)
            else:
                worst_r = max(worst_r, np.abs(dd).max() / M)
    assert worst_o < 1.0 and worst_r < 1.0


def test_one_period_lemma_on_random_states():
    """For filtered controls: h(t_k + s) >= h(t_k) (1 - alpha s) >= 0 over the whole period.

    This checks the implementation against the lemma in docs/CBF_DESIGN.md. It is a numerical
    check of one step at many states, not a proof."""
    fc, _ = make_cfgs()
    filt = CBFQPFilter(fc)
    rng = np.random.default_rng(3)
    B, N, K, m = 400, 3, 4, 200
    checked_o = checked_r = tight = 0
    for rep in range(12):
        # random states with every barrier >= 0
        poses = np.concatenate([rng.uniform(-0.8, 0.8, (B, N, 2)), rng.uniform(-np.pi, np.pi, (B, N, 1))], 2)
        obst = rng.uniform(-0.8, 0.8, (B, K, 2))
        h_o, h_r = barrier_values(fc, poses, obst)
        good = (h_o >= 0).all(axis=(1, 2)) & (h_r >= 0).all(axis=(1, 2))
        poses, obst = poses[good], obst[good]
        if len(poses) == 0:
            continue
        # adversarial nominal: bang-bang box corners
        nom = np.stack([rng.choice([fc.v_min, fc.v_max], poses.shape[:2]),
                        rng.choice([-fc.w_max, fc.w_max], poses.shape[:2])], 2)
        out = filt(poses, nom, obst)
        ok = ~out["infeasible"].any(axis=1)
        poses, obst, u = poses[ok], obst[ok], out["u"][ok]
        Bn = len(poses)
        traj = np.stack([arc_poses(poses[:, i], u[:, i, 0], u[:, i, 1], fc.dt, m) for i in range(N)], 1)  # [B,N,m+1,3]
        p = barrier_point(traj, fc.ell)                                   # [B,N,m+1,2]
        s = np.linspace(0, fc.dt, m + 1)
        for k in range(K):
            h = np.linalg.norm(p - obst[:, None, k:k + 1, None, :].reshape(Bn, 1, 1, 2), axis=3) ** 2 - fc.d_obst ** 2
            h0 = h[:, :, :1]
            in_range = np.sqrt(h0 + fc.d_obst ** 2) <= fc.sensing_range
            bound = h0 * (1 - fc.alpha * s)
            assert ((h >= bound - 1e-9) | ~in_range).all()
            assert ((h >= -1e-9) | ~in_range).all()
            checked_o += in_range.sum()
            tight += (in_range & (h.min(axis=2, keepdims=True) < 1e-3)).sum()
        for i in range(N):
            for j in range(i + 1, N):
                h = np.linalg.norm(p[:, i] - p[:, j], axis=2) ** 2 - fc.d_rr ** 2
                h0 = h[:, :1]
                in_range = np.sqrt(h0 + fc.d_rr ** 2) <= fc.comm_range
                assert ((h >= h0 * (1 - fc.alpha * s) - 1e-9) | ~in_range).all()
                assert ((h >= -1e-9) | ~in_range).all()
                checked_r += in_range.sum()
    assert checked_o > 2000 and checked_r > 500
    assert tight > 0       # some cases ran close to the boundary, so the check is not vacuous


def test_filter_config_validation():
    with pytest.raises(ValueError):
        make_cfgs(alpha=11.0)[0].validate()                 # alpha*dt > 1
    with pytest.raises(ValueError):
        make_cfgs(sensing_range=0.2)[0].validate()          # cannot see far enough for one period
    with pytest.raises(ValueError):
        make_cfgs(v_min=0.1)[0].validate()                  # stopping must be allowed
