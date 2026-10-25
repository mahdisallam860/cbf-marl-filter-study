"""Common-prefix formation metric, executed-period counting, secondary metrics and recording, on
constructed synthetic series and hand-built layouts (no study layouts)."""
import numpy as np

from cbf_study.cbf_qp import barrier_values
from cbf_study.envs import KinematicEnv
from cbf_study.protocol import Outcome, formation_error, run_trials
from cbf_study.scenario import Layout
from cbf_study.trial_metrics import (REASON_BOTH_ZERO, REASON_ON_ZERO, RecordingEnv, both_goal_full_trajectory,
                                     common_prefix, executed_periods, formation_series, full_trajectory_mean,
                                     reached_goal_region, sampled_barrier_minima)
from conftest import make_cfgs

G, C, I, TO = Outcome.GOAL, Outcome.COLLISION, Outcome.INFEASIBLE, Outcome.TIMEOUT


def const_policy(v, w=0.0):
    return lambda poses, layout: np.tile([v, w], poses.shape[:2] + (1,)).astype(float)


def series(*cols, S=10):
    """[S+1, B] from per-trial lists (padded with a large 'frozen' value to catch misuse)."""
    out = np.full((S + 1, len(cols)), 99.0)
    for b, c in enumerate(cols):
        out[:len(c), b] = c
    return out


def test_identical_prefixes_give_zero():
    e = series([0.5, 0.1, 0.2, 0.3], [0.0, 0.04])
    d, L, why = common_prefix(e, [3, 1], e.copy(), [3, 1])
    assert np.array_equal(d, [0.0, 0.0]) and list(L) == [3, 1] and list(why) == ["", ""]


def test_only_the_common_executed_prefix_is_used_and_t0_and_frozen_samples_are_excluded():
    off = series([7.0, 0.1, 0.1, 0.1])                    # off ran 3 periods; index 0 is the initial state
    on = series([-7.0, 0.2, 0.3, 0.4, 0.5, 0.6])          # on ran 5 periods
    d, L, _ = common_prefix(off, [3], on, [5])
    assert L[0] == 3 and np.isclose(d[0], np.mean([0.1, 0.2, 0.3]))   # t = 1..3 only (not t = 0, not t > 3)


def test_infeasible_unexecuted_period_and_zero_prefix_excluded_never_zero():
    off = series([0.0, 0.1, 0.1, 0.1, 0.1])
    on = series([0.0, 0.2, 0.2])                          # INFEASIBLE at attempted period 3 -> T_on = 2
    d, L, _ = common_prefix(off, [4], on, [2])
    assert L[0] == 2 and np.isclose(d[0], 0.1)            # the frozen index 3 (value 99) is not used
    d, L, why = common_prefix(series([0.0], [0.0]), [5, 0], series([0.0], [0.0]), [0, 0])
    assert np.isnan(d).all() and list(L) == [0, 0]
    assert why[0] == REASON_ON_ZERO and why[1] == REASON_BOTH_ZERO


def test_recorded_trajectory_counts_collision_period_and_infeasible_attempt_correctly():
    fc, pc = make_cfgs(max_steps=40)
    # trial: robot drives into an obstacle -> COLLISION; the colliding period is executed (T = end_step)
    lay = Layout(np.array([[[-1.0, 0.0, 0.0]]]), np.array([[[0.0, 0.0]]]), np.array([[5.0, 0.0]]), np.zeros((1, 2)))
    env = RecordingEnv(KinematicEnv(pc.dt))
    res = run_trials(env, const_policy(0.5), lay, pc, fc, use_filter=False)
    T = executed_periods(res)[0]
    traj = env.trajectory(pc.max_steps)
    assert res.outcome[0] == C and T == res.end_step[0] > 0
    d_T = np.linalg.norm(traj[T, 0, 0, :2] - lay.obst_xy[0, 0]) - (pc.robot_r + pc.obst_r)
    d_prev = np.linalg.norm(traj[T - 1, 0, 0, :2] - lay.obst_xy[0, 0]) - (pc.robot_r + pc.obst_r)
    assert d_prev > 0.01 and d_T <= 1e-9                       # index T is the collision period's end state (contact)
    assert np.allclose(traj[T, 0], res.final_poses[0])
    assert np.allclose(traj[T + 1:], traj[T])                  # frozen afterwards
    # pinched robot with the filter on: INFEASIBLE at period 0, nothing executed (T = 0)
    D = fc.d_obst
    x = np.array([-fc.ell, 0.0])
    lay2 = Layout(np.array([[[x[0], x[1], 0.0]]]), np.array([[[D + 1e-3, 0.0], [-D - 1e-3, 0.0]]]),
                  np.array([[5.0, 0.0]]), np.zeros((1, 2)))
    env2 = RecordingEnv(KinematicEnv(pc.dt))
    on = run_trials(env2, const_policy(0.0), lay2, pc, fc, use_filter=True)
    assert on.outcome[0] == I and executed_periods(on)[0] == 0
    assert on.step_log["active"][0, 0]                          # 'active' is set although nothing executed


def test_formation_series_matches_protocol_definition():
    fc, pc = make_cfgs(max_steps=5)
    slots = np.array([[0.0, 0.3], [-0.26, -0.15], [0.26, -0.15]])
    init = np.array([[[0.0, 0.32, 0.0], [-0.25, -0.15, 0.1], [0.27, -0.12, -0.1]]])
    lay = Layout(init, np.zeros((1, 0, 2)), np.array([[3.0, 0.0]]), slots)
    env = RecordingEnv(KinematicEnv(pc.dt))
    res = run_trials(env, const_policy(0.3, 0.2), lay, pc, fc, use_filter=False)
    e = formation_series(env.trajectory(pc.max_steps), slots)
    assert np.isclose(e[res.end_step[0], 0], res.form_err_final[0])
    assert np.isclose(e[0, 0], formation_error(init, slots)[0])


def test_full_trajectory_secondary_uses_all_executed_periods_and_only_joint_goal_pairs():
    e_off = series([9.0, 0.1, 0.1, 0.1], [9.0, 0.1], [9.0, 0.2, 0.2])
    e_on = series([9.0, 0.2, 0.2, 0.2, 0.2, 0.5], [9.0, 0.3, 0.3], [9.0, 0.4])
    both, m_off, m_on = both_goal_full_trajectory(e_off, [3, 1, 2], [G, G, C], e_on, [5, 2, 1], [G, TO, G])
    assert list(both) == [True, False, False]
    assert np.isclose(m_off[0], 0.1) and np.isclose(m_on[0], np.mean([0.2, 0.2, 0.2, 0.2, 0.5]))
    assert np.isnan(m_off[1:]).all() and np.isnan(m_on[1:]).all()
    assert np.isnan(full_trajectory_mean(e_off, [0, 1, 2])[0])


def test_reached_goal_region_ignores_the_collision_end_state():
    cd = series([1.0, 0.5, 0.25], [1.0, 0.5, 0.25], [0.2])
    r = reached_goal_region(cd, [2, 2, 0], [TO, C, I])
    assert list(r) == [True, False, True]                     # collision end state is not "before" the collision


def test_sampled_barrier_minima_include_terminal_state():
    fc, _ = make_cfgs()
    traj = np.zeros((4, 1, 1, 3))
    traj[:, 0, 0, 0] = [-2.0, -1.5, -1.0, -0.6]               # approaching an obstacle at the origin
    obst = np.array([[[0.0, 0.0]]])
    ho, _ = sampled_barrier_minima(traj, np.array([3]), obst, fc)
    h_last, _ = barrier_values(fc, traj[3], obst)
    assert np.isclose(ho[0], h_last.min())                     # terminal state counted
    ho2, _ = sampled_barrier_minima(traj, np.array([1]), obst, fc)
    assert ho2[0] > ho[0]                                      # states after T are not
