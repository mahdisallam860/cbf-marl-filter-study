"""B-match comparator on constructed synthetic states and layouts (no study layouts, no policy
checkpoints, no sweeps)."""
import numpy as np
import pytest

import cbf_study.cbf_qp as cbf_qp
import cbf_study.qp2d as qp2d
from cbf_study.bmatch import BMatchBrake, barrier_rows, run_bmatch
from cbf_study.cbf_qp import CBFQPFilter
from cbf_study.envs import KinematicEnv
from cbf_study.protocol import Outcome, run_trials
from cbf_study.qp2d import STATUS_NOMINAL
from cbf_study.scenario import Layout
from conftest import make_cfgs


def const_policy(v, w=0.0):
    return lambda poses, layout: np.tile([v, w], poses.shape[:2] + (1,)).astype(float)


def layout1(init, obst, goal=(10.0, 0.0)):
    init = np.asarray(init, float)
    obst = np.asarray(obst, float).reshape(init.shape[0], -1, 2)
    return Layout(init, obst, np.tile(goal, (init.shape[0], 1)), np.zeros((init.shape[1], 2)))


def test_admissible_clipped_command_unchanged():
    fc, _ = make_cfgs()
    poses = np.array([[[0.0, 0.0, 0.0]]])
    obst = np.array([[[5.0, 5.0]]])                                  # far outside sensing range
    out = BMatchBrake(fc)(poses, np.array([[[0.9, -3.0]]]), obst)   # outside the box -> clipped only
    assert not out["activated"][0, 0]
    assert np.allclose(out["u"], [[[0.5, -1.0]]]) and np.allclose(out["u"], out["u_nom_clip"])


def test_triggered_brake_zeroes_v_and_keeps_clipped_w():
    fc, _ = make_cfgs()
    x = np.array([0.0, 0.0])
    c = x + np.array([fc.d_obst + 0.01, 0.0])                         # obstacle just ahead, inside the cover margin
    poses = np.array([[[x[0], x[1], 0.0]]])
    out = BMatchBrake(fc)(poses, np.array([[[0.5, 2.5]]]), c[None, None])
    assert out["activated"][0, 0]
    assert out["u"][0, 0, 0] == 0.0 and out["u"][0, 0, 1] == 1.0     # clipped w kept


@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 5.0, 10.0])
def test_rows_match_the_filter_at_the_passed_gain(alpha):
    """B-match triggers exactly when the filter, given the clipped command, cannot return it unchanged."""
    fc, _ = make_cfgs(alpha=alpha)
    rng = np.random.default_rng(0)
    B, N, K = 400, 3, 3
    cen = rng.uniform(-0.5, 0.5, (B, 1, 2))
    poses = np.concatenate([cen + rng.uniform(-0.45, 0.45, (B, N, 2)), rng.uniform(-np.pi, np.pi, (B, N, 1))], axis=2)
    obst = rng.uniform(-1.2, 1.2, (B, K, 2))
    u = np.stack([rng.uniform(-0.7, 0.7, (B, N)), rng.uniform(-1.5, 1.5, (B, N))], axis=2)
    bm = BMatchBrake(fc)(poses, u, obst)
    filt = CBFQPFilter(fc)(poses, bm["u_nom_clip"], obst)
    assert (bm["activated"] == (filt["status"] != STATUS_NOMINAL)).all()
    assert 0 < bm["activated"].mean() < 1                            # both branches exercised
    assert (bm["n_rows"] == filt["n_rows"]).all()                    # same enabled rows (range masks)


def test_range_masks_and_robot_rows_depend_on_gain():
    fc, _ = make_cfgs(alpha=0.5)
    # obstacle barrier-point distance just inside / just outside the sensing range
    for dist, enabled in ((fc.sensing_range - 1e-6, True), (fc.sensing_range + 1e-6, False)):
        poses = np.array([[[0.0, 0.0, 0.0]]])
        obst = np.array([[[fc.ell + dist, 0.0]]])
        _, _, m = barrier_rows(fc, poses, obst)
        assert m[0, 0, 0] == enabled
    # neighbour just inside / outside the communication range (equal headings: barrier = centre spacing)
    for s, enabled in ((fc.comm_range - 1e-6, True), (fc.comm_range + 1e-6, False)):
        poses = np.array([[[0.0, 0.0, 0.0], [0.0, s, 0.0]]])
        _, _, m = barrier_rows(fc, poses, np.zeros((1, 0, 2)))
        assert m[0, 0, 1] == enabled and m[0, 1, 0] == enabled and not m[0, 0, 0]
    # a pair 0.60 m apart, both commanded to stay put: the zero command violates the robot row at
    # alpha = 0.5 (needs >= 0.668 m) but not at alpha = 1 (needs >= 0.501 m)
    poses = np.array([[[0.0, 0.0, 0.0], [0.0, 0.60, 0.0]]])
    zero = np.zeros((1, 2, 2))
    a05 = BMatchBrake(make_cfgs(alpha=0.5)[0])(poses, zero, np.zeros((1, 0, 2)))
    a1 = BMatchBrake(make_cfgs(alpha=1.0)[0])(poses, zero, np.zeros((1, 0, 2)))
    assert a05["activated"].all() and not a1["activated"].any()


def test_bmatch_never_calls_the_qp_solver(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("QP solver called")
    monkeypatch.setattr(qp2d, "solve_qp2d", boom)
    monkeypatch.setattr(cbf_qp, "solve_qp2d", boom)
    fc, pc = make_cfgs(max_steps=60)
    lay = layout1([[[-1.0, 0.0, 0.0]]], [[0.0, 0.0]])
    run = run_bmatch(KinematicEnv(pc.dt), const_policy(0.5, 0.3), lay, pc, fc)
    assert run.n_brake_activations[0] > 0


def test_logging_separates_actor_command_applied_command_and_brake():
    fc, pc = make_cfgs(max_steps=80)
    lay = layout1([[[-1.0, 0.0, 0.0]], [[-1.0, 3.0, 0.0]]], [[0.0, 0.0], [0.0, 9.0]])
    run = run_bmatch(KinematicEnv(pc.dt), const_policy(0.8, -0.4), lay, pc, fc)
    res, L, ex = run.res, run.res.step_log, run.executed
    # the actor's own output is kept, even in braked periods
    assert np.allclose(run.actor_raw[ex], [0.8, -0.4]) and np.allclose(L["raw"][ex], [0.8, -0.4])
    assert np.allclose(run.actor_clipped[ex], [0.5, -0.4])
    br = run.brake_active
    assert br[:, 0].any() and not br[:, 1].any()                      # trial 1 never near an obstacle
    assert np.allclose(run.applied[br], [0.0, -0.4])                   # braked: v = 0, clipped w
    assert np.allclose(run.applied[ex & ~br.any(axis=2)][..., 0], 0.5)
    assert np.allclose(L["bmatch_command"][ex], run.applied[ex])       # relabelled, not 'raw'
    assert (res.n_interventions == br.sum(axis=(0, 2))).all() and res.n_interventions[0] > 0
    assert (L["status"][ex] == -1).all()                              # 'na': no QP was solved
    assert np.isnan(run.actor_raw[~ex]).all() and not br[~ex].any()   # nothing logged past the end


def test_unchanged_protocol_semantics_when_brake_never_triggers():
    fc, pc = make_cfgs(max_steps=50)
    lay = layout1([[[-1.0, 0.0, 0.0]]], [[0.0, 5.0]], goal=(0.5, 0.0))
    off = run_trials(KinematicEnv(pc.dt), const_policy(0.5), lay, pc, fc, use_filter=False)
    bm = run_bmatch(KinematicEnv(pc.dt), const_policy(0.5), lay, pc, fc)
    assert bm.n_brake_activations[0] == 0
    for k in ("outcome", "end_step", "collision_kind", "final_poses", "goal_dist_final"):
        assert np.array_equal(getattr(off, k), getattr(bm.res, k))
    assert off.outcome[0] == Outcome.GOAL
