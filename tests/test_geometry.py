import numpy as np

from cbf_study.geometry import (
    arc_poses, endpoint_sep_obstacle, endpoint_sep_robots, exact_arc_point_distance,
    swept_sep_lb_obstacle, swept_sep_lb_robots,
)

# Constructed cases use small bodies and a large step so a single step is longer than the
# collision diameter. These numbers are test fixtures, not study parameters.
R_ROBOT, R_OBST = 0.02, 0.01
R_SUM = R_ROBOT + R_OBST            # 0.03
V, DT = 1.0, 0.1                    # step length 0.1 m  >  2 * R_SUM = 0.06


def test_straight_crossing_missed_by_endpoints_caught_by_swept():
    pose = np.array([[-0.05, 0.0, 0.0]])
    v, w = np.array([V]), np.array([0.0])
    centre = np.array([0.0, 0.0])
    assert endpoint_sep_obstacle(pose, v, w, DT, centre, R_SUM)[0] > 0        # endpoints: no overlap
    assert swept_sep_lb_obstacle(pose, v, w, DT, centre, R_SUM)[0] < 0        # swept: overlap found
    assert exact_arc_point_distance(pose[0], V, 0.0, DT, centre) < R_SUM      # oracle agrees


def test_curved_crossing_missed_by_endpoints_caught_by_swept():
    pose = np.array([[-0.05, 0.0, 0.2]])
    v, w = np.array([V]), np.array([3.0])
    mid = arc_poses(pose, v, w, DT, 2)[0, 1, :2]                  # place obstacle on the arc midpoint
    assert endpoint_sep_obstacle(pose, v, w, DT, mid, R_SUM)[0] > 0
    assert swept_sep_lb_obstacle(pose, v, w, DT, mid, R_SUM)[0] < 0
    assert exact_arc_point_distance(pose[0], V, 3.0, DT, mid) < 1e-9


def test_robot_robot_swap_missed_by_endpoints_caught_by_swept():
    pi = np.array([[-0.05, 0.0, 0.0]])
    pj = np.array([[0.05, 0.0, np.pi]])
    v = np.array([V]); w = np.array([0.0])
    r_sum = 2 * R_ROBOT
    assert endpoint_sep_robots(pi, v, w, pj, v, w, DT, r_sum)[0] > 0
    assert swept_sep_lb_robots(pi, v, w, pj, v, w, DT, r_sum)[0] < 0


def test_swept_bound_never_misses_and_is_tight():
    """Randomised check against the exact oracle: lower bound <= exact separation, within L*tau."""
    rng = np.random.default_rng(1)
    m = 64
    worst_gap = 0.0
    for _ in range(300):
        pose = np.array([[rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-np.pi, np.pi)]])
        v = rng.uniform(-0.5, 0.5); w = rng.uniform(-1.0, 1.0)
        c = rng.uniform(-0.3, 0.3, 2)
        exact = exact_arc_point_distance(pose[0], v, w, 0.1, c)
        lb = swept_sep_lb_obstacle(pose, np.array([v]), np.array([w]), 0.1, c, 0.0, m)[0]
        assert lb <= exact + 1e-12                     # conservative: never above the truth
        worst_gap = max(worst_gap, exact - lb)
    assert worst_gap <= 0.5 * (0.1 / m) + 1e-12 + 1e-9  # |v| <= 0.5, so gap <= |v| tau / 2 + sampling


def test_oracle_arc_matches_dense_sampling():
    pose = np.array([[0.0, 0.0, 0.3]])
    v, w, dt = 0.5, 1.0, 0.1
    c = np.array([0.02, 0.04])
    dense = arc_poses(pose, np.array([v]), np.array([w]), dt, 20000)[0, :, :2]
    brute = np.linalg.norm(dense - c, axis=1).min()
    assert abs(exact_arc_point_distance(pose[0], v, w, dt, c) - brute) < 1e-6
