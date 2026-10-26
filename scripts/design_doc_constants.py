#!/usr/bin/env python3
"""Analytic constants quoted in docs/CBF_DESIGN.md (sections 4a-4d). Arithmetic on config/study.toml
values only (no simulation). Checks the closed-form one-sided obstacle coefficient kappa_o(d)
(two branches) against a brute-force maximisation of F(w) = max_{0<=s<=V_p} [2(d + s dt) s w - 2 s^2]_+,
kappa_o(d) = dt F(w_max) / (2 w_max), so that the one-sided state-only obstacle margin is
m_B,o(d) = kappa_o(d) w_max = (dt/2) F(w_max); pair share kappa_r(d) = dt V_p (d + 2 V_p dt), m_B,r = kappa_r w_max.
Writes results/design/margin_constants.json."""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbf_study.cbf_qp import margin_obstacle, margin_robot   # noqa: E402
from cbf_study.config import StudyConfig                     # noqa: E402

cfg = StudyConfig.load()
fc = cfg.filter_config(10.0)
dt, w, Vp = fc.dt, fc.w_max, fc.vp
d_star = 2 * Vp * (1 - dt * w) / w


def kappa_o(d):
    return dt * d * d * w / (4 * (1 - dt * w)) if d <= d_star else (dt / w) * ((d + Vp * dt) * w * Vp - Vp ** 2)


def kappa_o_brute(d, n=200001):
    s = np.linspace(0.0, Vp, n)
    F = np.maximum(2 * (d + s * dt) * s * w - 2 * s * s, 0).max()
    return dt * F / (2 * w)


ds = np.r_[np.linspace(fc.d_obst, fc.sensing_range, 66), d_star]
err = max(abs(kappa_o(d) - kappa_o_brute(d)) for d in ds)
kr = lambda d: dt * Vp * (d + 2 * Vp * dt)
out = dict(
    note="analytic comparison only; the implemented filter (margins C) is unchanged",
    dt=dt, w_max=w, V_p=Vp, ell=fc.ell, D_o=fc.d_obst, D_rr=fc.d_rr, d_star=d_star,
    kappa_o_branch_check_max_abs_error=err,
    kappa_o_at={f"{d:.4f}": kappa_o(d) for d in (fc.d_obst, 0.5, d_star, fc.sensing_range)},
    kappa_o_branch2_at_d_star=(dt / w) * ((d_star + Vp * dt) * w * Vp - Vp ** 2),
    obstacle_margin_C_at_D_o=float(margin_obstacle(fc, fc.d_obst)), obstacle_margin_B_at_D_o=kappa_o(fc.d_obst) * w,
    pair_share_C_at_0p52=float(margin_robot(fc, 0.52)) / 2, pair_share_B_at_0p52=kr(0.52) * w,
    kappa_r_at={f"{d:.2f}": kr(d) for d in (fc.d_rr, 0.52, fc.comm_range)},
    standoff_hstar_alpha10_obstacle_at_D_o=float(margin_obstacle(fc, fc.d_obst)) / 10.0,
)
(ROOT / "results" / "design").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "design" / "margin_constants.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
assert err < 1e-6, err
