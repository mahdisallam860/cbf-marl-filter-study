import numpy as np
import pytest

from cbf_study.envs import KinematicEnv
from cbf_study.protocol import Outcome, run_trials
from cbf_study.scenario import SlotPolicy, sample_layout
from cbf_study.stress import run_stress, summarise
from conftest import make_cfgs

vmas = pytest.importorskip("vmas")
from cbf_study.envs import VmasEnv  # noqa: E402


def vmas_factory(fc, batch, n_obst, seed=0):
    return lambda: VmasEnv(batch, 3, n_obst, fc.dt, fc.robot_r, fc.obst_r, fc.v_max, fc.w_max, seed=seed)


def test_vmas_executes_the_same_motion_and_same_outcomes_as_exact_unicycle():
    fc, pc = make_cfgs(max_steps=200)
    B, K = 12, 4
    lay = sample_layout(11, B, K, fc)
    pol = SlotPolicy(fc.v_max, fc.w_max)
    for use in (False, True):
        k = run_trials(KinematicEnv(pc.dt), pol, lay, pc, fc, use_filter=use, log_steps=False)
        v = run_trials(vmas_factory(fc, B, K)(), pol, lay, pc, fc, use_filter=use, log_steps=False)
        assert v.track_err_max.max() < 1e-4                  # float32 simulator vs exact arc
        assert (k.outcome == v.outcome).all()
        assert (k.end_step == v.end_step).all()


def test_stress_filter_on_has_no_collisions_kinematic():
    rows = []
    for dt, alpha in ((0.1, 0.5), (0.1, 2.0), (0.1, 5.0), (0.05, 2.0), (0.05, 10.0)):
        fc, pc = make_cfgs(dt=dt, alpha=alpha, max_steps=int(25 / dt))
        for K in (2, 5, 8):
            for which in ("attack", "random"):
                off, on = run_stress(lambda: KinematicEnv(dt), pc, fc, K, 60, 100 + K, which)
                s = summarise(off, on)
                rows.append(s)
                assert s["on_collision"] == 0, (dt, alpha, K, which, s)
                assert s["on_h_obst_min"] >= -1e-9 and s["on_h_rr_min"] >= -1e-9, (dt, alpha, K, which, s)
    assert sum(r["off_collision"] for r in rows) > 100           # the baseline is genuinely unsafe here
    assert sum(r["on_interventions"] for r in rows) > 1000


def test_stress_filter_on_has_no_collisions_vmas():
    fc, pc = make_cfgs(max_steps=250)
    for K in (3, 6):
        for which in ("attack", "random"):
            off, on = run_stress(vmas_factory(fc, 40, K, seed=5), pc, fc, K, 40, 200 + K, which)
            s = summarise(off, on)
            assert s["on_collision"] == 0 and on.track_err_max.max() < 1e-4, s
            assert s["on_h_obst_min"] >= -1e-6 and s["on_h_rr_min"] >= -1e-6, s
