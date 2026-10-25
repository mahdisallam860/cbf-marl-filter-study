"""
Exact batched solver for the 2-variable QP used by the safety filter.

    minimise   0.5 * |u - u0|^2
    subject to a_k . u >= b_k   for every enabled row k

The feasible set is a convex polygon (possibly empty or unbounded). The closest point of a convex
polygon to u0 is one of:
  1. u0 itself (if feasible),
  2. the projection of u0 onto one constraint line, or
  3. a vertex (intersection of two constraint lines).
The solver builds all such candidates, discards those that violate any row, and returns the
nearest. This is exact up to the stated tolerance, is vectorised over a batch, and has no
iteration limit, so the filter's "infeasible" verdict is a property of the constraints, not of a
solver stopping early. OSQP is used only in the test-suite as an independent cross-check.

Rows are normalised to unit length, so `tol` is a distance in the units of u (m/s, rad/s).
"""
from __future__ import annotations

import numpy as np

# Solver status codes (also written to the step log)
STATUS_NOMINAL = 0      # nominal command already feasible, returned unchanged
STATUS_MODIFIED = 1     # nominal infeasible, closest feasible command returned
STATUS_INFEASIBLE = 2   # no command satisfies all rows


def solve_qp2d(u0, A, b, mask, tol=1e-9):
    """Solve a batch of 2-D projection QPs.

    u0   [B,2]    nominal command
    A    [B,K,2]  constraint normals
    b    [B,K]    constraint offsets
    mask [B,K]    True for rows that are enabled
    Returns (u [B,2], status [B] int). For STATUS_INFEASIBLE rows u is NaN.
    """
    u0 = np.asarray(u0, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    B, K, _ = A.shape

    nrm = np.linalg.norm(A, axis=2)
    degenerate = mask & (nrm < 1e-12)
    # A zero row reads 0 >= b: infeasible when b > tol, otherwise vacuous.
    degenerate_infeasible = (degenerate & (b > tol)).any(axis=1)
    ok = mask & ~degenerate
    safe = np.where(nrm < 1e-12, 1.0, nrm)
    An = A / safe[:, :, None]
    bn = b / safe

    # candidate 1: u0
    cands = [u0[:, None, :]]
    valid = [np.ones((B, 1), dtype=bool)]

    # candidate 2: projection of u0 onto each constraint line
    slack = bn - np.einsum("bkd,bd->bk", An, u0)               # >0 means u0 violates the row
    cands.append(u0[:, None, :] + slack[:, :, None] * An)
    valid.append(ok)

    # candidate 3: pairwise intersections
    ii, jj = np.triu_indices(K, 1)
    if len(ii):
        ai, aj = An[:, ii, :], An[:, jj, :]
        det = ai[:, :, 0] * aj[:, :, 1] - ai[:, :, 1] * aj[:, :, 0]
        good = ok[:, ii] & ok[:, jj] & (np.abs(det) > 1e-9)
        sd = np.where(good, det, 1.0)
        u1 = (bn[:, ii] * aj[:, :, 1] - ai[:, :, 1] * bn[:, jj]) / sd
        u2 = (ai[:, :, 0] * bn[:, jj] - aj[:, :, 0] * bn[:, ii]) / sd
        cands.append(np.stack([u1, u2], axis=2))
        valid.append(good)

    cand = np.concatenate(cands, axis=1)                        # [B,C,2]
    valid = np.concatenate(valid, axis=1)                       # [B,C]

    viol = bn[:, None, :] - np.einsum("bcd,bkd->bck", cand, An)  # [B,C,K]
    feasible = valid & ~((viol > tol) & ok[:, None, :]).any(axis=2)
    cost = np.where(feasible, ((cand - u0[:, None, :]) ** 2).sum(axis=2), np.inf)
    best = cost.argmin(axis=1)
    found = np.isfinite(cost[np.arange(B), best]) & ~degenerate_infeasible

    u = cand[np.arange(B), best]
    u = np.where(found[:, None], u, np.nan)
    unchanged = found & (best == 0)
    status = np.where(~found, STATUS_INFEASIBLE, np.where(unchanged, STATUS_NOMINAL, STATUS_MODIFIED))
    return u, status.astype(np.int64)
