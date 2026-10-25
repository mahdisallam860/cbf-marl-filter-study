"""Evaluation plumbing: filter-off does not depend on alpha (so one filter-off run is paired with
every filter-on run), pilot layouts are fixed by seed, and the pilot refuses to run without a
passed gate."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from cbf_study.config import StudyConfig
from cbf_study.envs import KinematicEnv
from cbf_study.evaluate import eval_layout, same_layouts
from cbf_study.policies import FrozenMAPPOPolicy
from cbf_study.protocol import run_trials
from test_policies import random_checkpoint

CFG = StudyConfig.load()
ROOT = Path(__file__).resolve().parents[1]


def test_filter_off_is_independent_of_alpha(tmp_path):
    pol = FrozenMAPPOPolicy(random_checkpoint(tmp_path / "c.pt"))
    lay = eval_layout(CFG, 5, "pilot", 6)
    pc = CFG.protocol_config()
    a = run_trials(KinematicEnv(pc.dt), pol, lay, pc, CFG.filter_config(CFG.alphas[0]), use_filter=False)
    b = run_trials(KinematicEnv(pc.dt), pol, lay, pc, CFG.filter_config(CFG.alphas[-1]), use_filter=False)
    assert np.array_equal(a.outcome, b.outcome) and np.array_equal(a.final_poses, b.final_poses)
    assert np.array_equal(a.h_obst_min, b.h_obst_min)
    assert np.allclose(a.step_log["applied"], b.step_log["applied"], equal_nan=True)


def test_pilot_and_validation_layouts_are_fixed_and_distinct():
    for K in CFG.densities:
        assert same_layouts(eval_layout(CFG, K, "pilot", 4), eval_layout(CFG, K, "pilot", 4))
        assert not same_layouts(eval_layout(CFG, K, "pilot", 4), eval_layout(CFG, K, "validation", 4))


def test_pilot_refuses_without_a_passed_gate(tmp_path):
    ck = str(random_checkpoint(tmp_path / "c.pt"))
    gate = dict(best=dict(ckpt=ck, gate_pass=False), best_per_seed={"0": dict(ckpt=ck, gate_pass=False)})
    (tmp_path / "sel.json").write_text(json.dumps(gate))
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_pilot.py"), "--ckpt", ck, "--tag", "t",
                        "--gate", str(tmp_path / "sel.json"), "--out", str(tmp_path / "out")],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode != 0 and "STOP" in r.stderr
    assert not (tmp_path / "out").exists()
