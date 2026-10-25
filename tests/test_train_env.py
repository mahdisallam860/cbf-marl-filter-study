"""Training environment = evaluation environment: same VMAS scenario, same layouts, same geometric
scoring and stop-at-first-collision rule, observations from (poses, layout) only."""
import numpy as np
import pytest

from cbf_study.config import StudyConfig
from cbf_study.observations import build_obs, obs_dim
from cbf_study.protocol import Outcome, run_trials
from cbf_study.scenario import Layout, SlotPolicy, sample_layout

pytest.importorskip("vmas")
from cbf_study.envs import VmasEnv                                   # noqa: E402
from cbf_study.train_env import FormationTrainEnv, TrainVmasEnv     # noqa: E402

CFG = StudyConfig.load()


def rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def test_obs_shape_padding_and_range():
    oc = CFG.obs_config()
    fc = CFG.layout_filter_config()
    for K in (0, 2, 8):
        lay = sample_layout(3, 4, K, fc, CFG.scenario_config())
        o = build_obs(lay.init_poses, lay, oc)
        assert o.shape == (4, 3, obs_dim(oc)) and o.dtype == np.float32 and np.isfinite(o).all()
        flags = o[..., 4 + 4 * 2:4 + 4 * 2 + 4 * oc.k_obst].reshape(4, 3, oc.k_obst, 4)[..., 3]
        assert (flags.sum(axis=2) <= min(K, oc.k_obst)).all()
        if K == 0:
            assert (flags == 0).all()
    # an obstacle beyond obs_range is reported absent
    lay = Layout(np.array([[[0, 0, 0], [0, 1, 0], [1, 0, 0]]], float), np.array([[[50.0, 0.0]]]),
                 np.array([[3.0, 0.0]]), np.zeros((3, 2)))
    o = build_obs(lay.init_poses, lay, oc).reshape(1, 3, -1)
    assert (o[..., 12 + 3] == 0).all() and np.allclose(o[..., 12 + 2], oc.obs_range)


def test_obs_invariant_to_rigid_motion_of_the_whole_scene():
    """Body-frame features: rotating and translating poses, obstacles, goal and slots together
    leaves the observation unchanged (up to float32 rounding)."""
    oc = CFG.obs_config()
    lay = sample_layout(5, 6, 5, CFG.layout_filter_config(), CFG.scenario_config())
    o0 = build_obs(lay.init_poses, lay, oc)
    Rm, t, a = rot(0.7), np.array([0.3, -2.0]), 0.7
    P = lay.init_poses.copy()
    P[..., :2] = P[..., :2] @ Rm.T + t
    P[..., 2] += a
    lay2 = Layout(P, lay.obst_xy @ Rm.T + t, lay.goal @ Rm.T + t, lay.slots @ Rm.T)
    o1 = build_obs(P, lay2, oc)
    assert np.allclose(o0, o1, atol=1e-5)


def test_partial_reset_leaves_other_envs_bit_identical():
    fc = CFG.layout_filter_config()
    B, K = 8, 3
    lay = sample_layout(9, B, K, fc, CFG.scenario_config())
    other = sample_layout(10, B, K, fc, CFG.scenario_config())
    pol = SlotPolicy(fc.v_max, fc.w_max)
    a = TrainVmasEnv(B, 3, K, fc.dt, fc.robot_r, fc.obst_r, fc.v_max, fc.w_max, seed=0)
    b = TrainVmasEnv(B, 3, K, fc.dt, fc.robot_r, fc.obst_r, fc.v_max, fc.w_max, seed=0)
    a.reset(lay.init_poses, lay.obst_xy); b.reset(lay.init_poses, lay.obst_xy)
    mask = np.zeros(B, bool); mask[[1, 4]] = True
    for k in range(30):
        u = pol(a.poses(), lay)
        a.step(u); b.step(u)
        if k == 10:
            b.reset_subset(mask, np.where(mask[:, None, None], other.init_poses, b.poses()),
                           np.where(mask[:, None, None], other.obst_xy, lay.obst_xy))
            assert np.array_equal(b.poses()[mask], other.init_poses[mask].astype(np.float32).astype(np.float64))
    assert np.array_equal(a.poses()[~mask], b.poses()[~mask])


def test_training_env_reproduces_protocol_outcomes():
    """Same layouts and the same deterministic policy: first-episode outcome and end step in the
    training env equal protocol.run_trials (filter off, VmasEnv) for every trial."""
    pc, fc = CFG.protocol_config(), CFG.layout_filter_config()
    B, K = 24, 2
    lay = sample_layout(11, B, K, fc, CFG.scenario_config())
    pol = SlotPolicy(fc.v_max, fc.w_max)
    ref = run_trials(VmasEnv(B, 3, K, fc.dt, fc.robot_r, fc.obst_r, fc.v_max, fc.w_max, seed=0), pol, lay, pc, fc,
                     use_filter=False, log_steps=False)
    assert {Outcome.GOAL, Outcome.COLLISION} <= set(ref.outcome.tolist())

    env = FormationTrainEnv(CFG, run_seed=0, envs_per_density=B, densities=(K,))
    env.reset([lay])
    g = env.groups[0]
    first_out = np.zeros(B, int); first_len = np.zeros(B, int); first_ret = np.zeros(B)
    for k in range(pc.max_steps):
        live = Layout(g.layout.init_poses, g.layout.obst_xy, g.layout.goal, g.layout.slots)
        so = env.step(pol(g.poses, live))
        done = so.terminated | so.truncated
        for ep, b in zip(so.episodes, np.flatnonzero(done)):
            if first_out[b] == 0:
                first_out[b], first_len[b], first_ret[b] = ep["outcome"], ep["length"], ep["ret"]
        if (first_out > 0).all():
            break
    assert np.array_equal(first_out, ref.outcome)
    assert np.array_equal(first_len, ref.end_step)
    # collision penalty only on collided episodes; progress reward bounded by the start distance
    d0 = np.linalg.norm(lay.init_poses[..., :2].mean(axis=1) - lay.goal, axis=1)
    rc = CFG.reward_config()
    assert (first_ret <= rc.w_progress * d0 + 1e-6).all()
    col = first_out == Outcome.COLLISION
    assert (first_ret[col] <= rc.w_progress * d0[col] - min(rc.w_collision_obst, rc.w_collision_rr) + 1e-6).all()


def test_training_env_mixes_densities_and_never_reuses_eval_layouts():
    env = FormationTrainEnv(CFG, run_seed=1, envs_per_density=4)
    obs = env.reset()
    assert obs.shape[0] == 4 * len(CFG.densities)
    assert [g.K for g in env.groups] == list(CFG.densities)
    for g in env.groups:
        val = sample_layout(CFG.validation_seed(g.K), 4, g.K, CFG.layout_filter_config(), CFG.scenario_config())
        pil = sample_layout(CFG.pilot_seed(g.K), 4, g.K, CFG.layout_filter_config(), CFG.scenario_config())
        assert not np.allclose(g.layout.init_poses, val.init_poses)
        assert not np.allclose(g.layout.init_poses, pil.init_poses)
    rng = np.random.default_rng(0)
    for _ in range(5):
        so = env.step(rng.uniform(-1, 1, (env.n_envs, 3, 2)))
        assert so.obs.shape == obs.shape and so.reward.shape == (env.n_envs,)
