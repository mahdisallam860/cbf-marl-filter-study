import numpy as np
import pytest
from scipy import sparse

from cbf_study.qp2d import STATUS_INFEASIBLE, STATUS_MODIFIED, STATUS_NOMINAL, solve_qp2d

osqp = pytest.importorskip("osqp")


def osqp_solve(u0, A, b):
    prob = osqp.OSQP()
    prob.setup(P=sparse.eye(2, format="csc"), q=-u0, A=sparse.csc_matrix(A), l=b,
               u=np.full(len(b), np.inf), verbose=False, eps_abs=1e-10, eps_rel=1e-10,
               polishing=True, max_iter=200000)
    res = prob.solve()
    return res.info.status, res.x


def test_matches_osqp_on_feasible_problems():
    rng = np.random.default_rng(0)
    n_mod = 0
    for _ in range(400):
        K = rng.integers(2, 9)
        A = rng.normal(size=(K, 2))
        A /= np.linalg.norm(A, axis=1, keepdims=True)
        ustar = rng.uniform(-1, 1, 2)
        b = A @ ustar - rng.uniform(0.0, 0.8, K)         # ustar is strictly feasible
        u0 = rng.uniform(-2, 2, 2)
        u, st = solve_qp2d(u0[None], A[None], b[None], np.ones((1, K), bool))
        status, x = osqp_solve(u0, A, b)
        assert status == "solved"
        assert st[0] in (STATUS_NOMINAL, STATUS_MODIFIED)
        assert np.allclose(u[0], x, atol=1e-5), (u[0], x)
        assert (A @ u[0] - b >= -1e-8).all()
        n_mod += st[0] == STATUS_MODIFIED
    assert n_mod > 100                                    # the test exercised active constraints


def test_infeasible_detected_and_agrees_with_osqp():
    rng = np.random.default_rng(1)
    for _ in range(100):
        a = rng.normal(size=2)
        a /= np.linalg.norm(a)
        gap = rng.uniform(0.05, 1.0)
        # a.u >= c and -a.u >= gap - c  ->  contradictory
        A = np.stack([a, -a, rng.normal(size=2)])
        c = rng.uniform(-1, 1)
        b = np.array([c, gap - c, -3.0])
        u, st = solve_qp2d(np.zeros((1, 2)), A[None], b[None], np.ones((1, 3), bool))
        status, _ = osqp_solve(np.zeros(2), A, b)
        assert st[0] == STATUS_INFEASIBLE and np.isnan(u).all()
        assert "infeasible" in status


def test_masked_rows_ignored_and_nominal_passthrough():
    A = np.array([[[1.0, 0.0], [-1.0, 0.0]]])
    b = np.array([[5.0, 5.0]])                              # contradictory, but both disabled
    u0 = np.array([[0.3, -0.2]])
    u, st = solve_qp2d(u0, A, b, np.zeros((1, 2), bool))
    assert st[0] == STATUS_NOMINAL and np.allclose(u, u0)


def test_batch_matches_single():
    rng = np.random.default_rng(2)
    B, K = 30, 5
    A = rng.normal(size=(B, K, 2))
    b = rng.uniform(-1, 0, size=(B, K))
    u0 = rng.uniform(-1, 1, size=(B, 2))
    m = rng.random((B, K)) < 0.8
    ub, sb = solve_qp2d(u0, A, b, m)
    for i in range(B):
        u1, s1 = solve_qp2d(u0[i:i+1], A[i:i+1], b[i:i+1], m[i:i+1])
        assert s1[0] == sb[i]
        assert np.allclose(u1[0], ub[i], equal_nan=True)
