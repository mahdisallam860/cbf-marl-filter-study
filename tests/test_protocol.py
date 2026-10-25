"""Protocol rules and the constructed cases requested for the study:
feasible approaches, constructed infeasible cases, and between-step crossings."""
import numpy as np
import pytest

from cbf_study.cbf_qp import CBFQPFilter, barrier_values
from cbf_study.envs import KinematicEnv
from cbf_study.geometry import swept_sep_lb_obstacle, endpoint_sep_obstacle
from cbf_study.protocol import Outcome, run_paired, run_trials, write_step_csv, write_summary_csv
from cbf_study.scenario import Layout, SlotPolicy, sample_layout
from conftest import make_cfgs

ENV = lambda dt: (lambda: KinematicEnv(dt))


def const_policy(v, w=0.0):
    def pol(poses, layout):
        out = np.zeros(poses.shape[:2] + (2,))
        out[..., 0], out[..., 1] = v, w
        return out
    return pol


def layout1(init, obst, goal=(10.0, 0.0), slots=None):
    init = np.asarray(init, float)           # [B,N,3]
    obst = np.asarray(obst, float).reshape(init.shape[0], -1, 2)
    B, N, _ = init.shape
    return Layout(init, obst, np.tile(goal, (B, 1)), np.zeros((N, 2)) if slots is None else slots)


# ---------------------------------------------------------------------------------------------
# Rule 2: stop at the first geometric collision; distinct outcomes
# ---------------------------------------------------------------------------------------------
def test_stop_at_first_collision_not_pass_through_and_complete():
    fc, pc = make_cfgs(max_steps=200, goal_tol=0.05)
    lay = layout1([[[-1.5, 0.0, 0.0]]], [[0.0, 0.0]], goal=(1.5, 0.0))
    pol = SlotPolicy(pc.v_max, pc.w_max)
    off = run_trials(KinematicEnv(pc.dt), pol, lay, pc, fc, use_filter=False)
    assert off.outcome[0] == Outcome.COLLISION and off.collision_kind[0] == 1
    # contact (separation 0) happens at x = -0.3, i.e. at the end of step 24 (0.05 m per step).
    # Touching is scored as a collision (conservative band < 1 mm), and the trial stops there.
    assert off.end_step[0] == 24
    assert off.final_poses[0, 0, 0] > -0.31 and off.final_poses[0, 0, 0] < 0.0
    # Without the stop rule the same robot would pass through the obstacle and "complete" the goal:
    env = KinematicEnv(pc.dt)
    env.reset(lay.init_poses, lay.obst_xy)
    for _ in range(pc.max_steps):
        env.step(np.clip(pol(env.poses(), lay), [pc.v_min, -pc.w_max], [pc.v_max, pc.w_max]))
    assert np.linalg.norm(env.poses()[0, 0, :2] - lay.goal[0]) < 0.05
    assert off.goal_dist_final[0] > 1.0


def test_four_outcomes_are_distinct_and_failures_are_counted():
    fc, pc = make_cfgs(max_steps=100, goal_tol=0.05)
    pol = SlotPolicy(pc.v_max, pc.w_max)
    # trial 0 free path -> GOAL ; trial 1 obstacle on the path -> COLLISION ; trial 2 stays put -> TIMEOUT
    lay = layout1([[[-1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]]],
                  [[0.0, 3.0], [0.0, 0.0], [0.0, 3.0]], goal=(1.0, 0.0))
    class Mixed:
        def __call__(self, poses, layout):
            a = pol(poses, layout)
            a[2] = 0.0
            return a
    r = run_trials(KinematicEnv(pc.dt), Mixed(), lay, pc, fc, use_filter=False)
    assert list(r.outcome) == [Outcome.GOAL, Outcome.COLLISION, Outcome.TIMEOUT]
    assert list(r.success) == [True, False, False]
    # INFEASIBLE on its own batch (constructed case below); here only check it is a separate code
    assert len({o for o in Outcome}) == 5


def test_identical_rules_when_filter_is_inactive():
    """No obstacle or neighbour in range: filter-on must reproduce filter-off exactly."""
    fc, pc = make_cfgs(max_steps=80, goal_tol=0.05)
    lay = layout1([[[-1.0, 0.0, 0.3]], [[-1.0, 0.5, -0.2]]], np.zeros((2, 0, 2)), goal=(1.0, 0.0))
    pol = SlotPolicy(pc.v_max, pc.w_max)
    off, on = run_paired(ENV(pc.dt), pol, lay, pc, fc)
    assert (off.outcome == on.outcome).all() and (off.end_step == on.end_step).all()
    assert np.allclose(off.final_poses, on.final_poses, atol=1e-12)
    assert on.n_interventions.sum() == 0
    assert np.allclose(off.step_log["applied"], on.step_log["applied"], equal_nan=True)


# ---------------------------------------------------------------------------------------------
# Rule 1: between-step crossing (constructed). Step length 0.1 m > collision diameter 0.06 m.
# ---------------------------------------------------------------------------------------------
TINY = dict(robot_r=0.02, obst_r=0.01, ell=0.01, v_min=-1.0, v_max=1.0, w_max=1.0, alpha=5.0,
            sensing_range=0.6, comm_range=0.6)


def test_between_step_crossing_scored_as_collision_by_swept_check_only():
    fc, pc = make_cfgs(**TINY, max_steps=3, goal_tol=0.01)
    lay = layout1([[[-0.05, 0.0, 0.0]]], [[0.0, 0.0]], goal=(0.05, 0.0))
    # The sampled poses straddle the obstacle without overlap ...
    pose0 = lay.init_poses[0]
    r_sum = pc.robot_r + pc.obst_r
    assert endpoint_sep_obstacle(pose0, np.array([1.0]), np.array([0.0]), pc.dt, lay.obst_xy[0, 0], r_sum)[0] > 0
    # ... so an endpoint-only scorer would call the next state safe and let the robot reach the goal:
    env = KinematicEnv(pc.dt)
    env.reset(lay.init_poses, lay.obst_xy)
    env.step(np.array([[[1.0, 0.0]]]))
    assert np.linalg.norm(env.poses()[0, 0, :2] - lay.goal[0]) < 1e-9
    assert np.linalg.norm(env.poses()[0, 0, :2] - lay.obst_xy[0, 0]) > r_sum
    # The protocol's swept check sees the crossing, and COLLISION wins over GOAL in the same period.
    off = run_trials(KinematicEnv(pc.dt), const_policy(1.0), lay, pc, fc, use_filter=False)
    assert off.outcome[0] == Outcome.COLLISION and off.end_step[0] == 1
    # swept bound is conservative: lower bound < 0 although both end points are clear
    lb = swept_sep_lb_obstacle(pose0, np.array([1.0]), np.array([0.0]), pc.dt, lay.obst_xy[0, 0], r_sum)
    assert lb[0] < 0


def test_between_step_crossing_filter_on_prevents_it():
    fc, pc = make_cfgs(**TINY, max_steps=60, goal_tol=0.01)
    offs = np.linspace(-0.04, 0.04, 9)
    init = np.array([[[-0.5, y, 0.0]] for y in offs])
    lay = layout1(init, np.zeros((len(offs), 1, 2)), goal=(0.5, 0.0))
    off, on = run_paired(ENV(pc.dt), const_policy(1.0), lay, pc, fc)
    assert (off.outcome[np.abs(offs) < 0.02] == Outcome.COLLISION).all()   # near-centre lines are hit
    assert not (on.outcome == Outcome.COLLISION).any()
    assert on.h_obst_min.min() >= -1e-9
    assert on.n_interventions.sum() > 0


# ---------------------------------------------------------------------------------------------
# Feasible approaches (study-like sizes)
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("heading", [0.0, 0.3, -0.5])
def test_feasible_robot_obstacle_approaches(heading):
    fc, pc = make_cfgs(max_steps=150)
    offs = np.linspace(-0.35, 0.35, 15)
    init = np.array([[[-1.0, y, heading]] for y in offs])
    lay = layout1(init, np.zeros((len(offs), 1, 2)), goal=(3.0, 0.0))
    off, on = run_paired(ENV(pc.dt), const_policy(pc.v_max), lay, pc, fc)
    assert (off.outcome == Outcome.COLLISION).sum() >= 3          # baseline does collide
    assert not (on.outcome == Outcome.COLLISION).any()
    assert not (on.outcome == Outcome.INFEASIBLE).any()
    assert on.h_obst_min.min() >= -1e-9
    assert on.n_interventions.sum() > 0
    # body clearance, from the realised final poses
    d = np.linalg.norm(on.final_poses[:, 0, :2], axis=1) - (pc.robot_r + pc.obst_r)
    assert (d > 0).all()


def test_feasible_robot_robot_head_on_and_crossing():
    fc, pc = make_cfgs(max_steps=150)
    ly = np.linspace(-0.15, 0.15, 7)
    # head-on with lateral offsets
    init = np.array([[[-1.0, 0.0, 0.0], [1.0, y, np.pi]] for y in ly])
    lay = layout1(init, np.zeros((len(ly), 0, 2)), goal=(10.0, 0.0), slots=np.zeros((2, 2)))
    off, on = run_paired(ENV(pc.dt), const_policy(pc.v_max), lay, pc, fc)
    assert (off.outcome == Outcome.COLLISION).any()
    assert (off.collision_kind[off.outcome == Outcome.COLLISION] == 2).all()
    assert not (on.outcome == Outcome.COLLISION).any() and not (on.outcome == Outcome.INFEASIBLE).any()
    assert on.h_rr_min.min() >= -1e-9
    # perpendicular crossing
    init = np.array([[[-1.0, 0.0, 0.0], [y, -1.0, np.pi / 2]] for y in ly - 0.0])
    lay = layout1(init, np.zeros((len(ly), 0, 2)), goal=(10.0, 0.0), slots=np.zeros((2, 2)))
    off, on = run_paired(ENV(pc.dt), const_policy(pc.v_max), lay, pc, fc)
    assert (off.outcome == Outcome.COLLISION).any()
    assert not (on.outcome == Outcome.COLLISION).any() and not (on.outcome == Outcome.INFEASIBLE).any()
    assert on.h_rr_min.min() >= -1e-9


# ---------------------------------------------------------------------------------------------
# Constructed infeasible cases: reported, counted as failures, never silently "solved"
# ---------------------------------------------------------------------------------------------
def test_constructed_infeasible_pinched_between_two_obstacles():
    fc, pc = make_cfgs(max_steps=20)
    D = fc.d_obst
    p = np.array([0.0, 0.0])                         # barrier point
    x = p - fc.ell * np.array([1.0, 0.0])            # axle centre behind it (heading 0)
    gap = 1e-3
    obst = [[p[0] + D + gap, 0.0], [p[0] - D - gap, 0.0]]       # one ahead, one behind, both just outside
    lay = layout1([[[x[0], x[1], 0.0]]], [obst], goal=(5.0, 0.0))
    # not a geometric collision: the robot is simply pinched
    assert np.linalg.norm(x - np.array(obst[1])) > pc.robot_r + pc.obst_r
    off = run_trials(KinematicEnv(pc.dt), const_policy(0.0), lay, pc, fc, use_filter=False)
    assert off.outcome[0] == Outcome.TIMEOUT                    # stationary robot is safe without the filter
    on = run_trials(KinematicEnv(pc.dt), const_policy(0.0), lay, pc, fc, use_filter=True)
    assert on.outcome[0] == Outcome.INFEASIBLE
    assert on.end_step[0] == 0 and on.n_infeasible[0] == 1       # stops before moving
    assert not on.success[0]
    assert on.step_log["status"][0, 0, 0] == 2 and on.step_log["infeasible"][0, 0, 0]
    assert np.isnan(on.step_log["applied"][0, 0]).all()
    # solver verdict agrees with an independent check: the two rows contradict (g_f = -g_b, delta > 0)
    out = CBFQPFilter(fc)(lay.init_poses, np.zeros((1, 1, 2)), lay.obst_xy)
    assert out["infeasible"][0, 0]


def test_constructed_infeasible_from_velocity_limits_no_reverse():
    # obstacle just outside the cover straight ahead; escaping needs v < 0
    base = dict(max_steps=10)
    fc, pc = make_cfgs(**base, v_min=0.0)
    D = fc.d_obst
    lay = layout1([[[-D - fc.ell - 1e-3, 0.0, 0.0]]], [[0.0, 0.0]], goal=(5.0, 0.0))
    on = run_trials(KinematicEnv(pc.dt), const_policy(0.5), lay, pc, fc, use_filter=True)
    assert on.outcome[0] == Outcome.INFEASIBLE
    # same geometry with reversing allowed is feasible and the filter backs the robot off
    fc2, pc2 = make_cfgs(**base, v_min=-0.5)
    on2 = run_trials(KinematicEnv(pc2.dt), const_policy(0.5), lay, pc2, fc2, use_filter=True)
    assert on2.outcome[0] != Outcome.INFEASIBLE and on2.n_interventions[0] > 0


# ---------------------------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------------------------
def test_logs_contain_required_fields(tmp_path):
    fc, pc = make_cfgs(max_steps=40)
    lay = sample_layout(7, 4, 3, fc)
    off, on = run_paired(ENV(pc.dt), SlotPolicy(pc.v_max, pc.w_max), lay, pc, fc)
    write_summary_csv(tmp_path / "s.csv", on, "t")
    write_step_csv(tmp_path / "steps.csv", on, trials=[0], label="t")
    head = (tmp_path / "steps.csv").read_text().splitlines()[0].split(",")
    for col in ("nom_v", "app_v", "h_obst_min", "h_rr_min", "intervened", "status", "infeasible"):
        assert col in head
    assert "n_infeasible" in (tmp_path / "s.csv").read_text().splitlines()[0]
    assert set(np.unique(off.step_log["status"][off.step_log["active"][:, :, None].repeat(3, 2)])) == {-1}
