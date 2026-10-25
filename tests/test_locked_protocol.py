"""The author-locked evaluation protocol leaves the training config and the actors untouched, keeps
author locking separate from advisor approval, and its analytical prediction follows from the
unchanged margin functions. No layout, policy rollout or filter trial is run here."""
import json
import tomllib
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from cbf_study.cbf_qp import margin_robot
from cbf_study.config import StudyConfig

ROOT = Path(__file__).resolve().parents[1]
LOCK = tomllib.loads((ROOT / "config" / "evaluation_protocol_locked.toml").read_text())
CFG = StudyConfig.load()


def test_training_config_and_actor_fingerprint_unchanged():
    assert CFG.env_fingerprint() == LOCK["lock"]["env_fingerprint"] == "c60352aee3f11ed8"
    assert CFG.densities == (2, 5, 8)                       # training mixture, not the evaluation roles
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())
    assert sel["env_fingerprint"] == CFG.env_fingerprint()
    assert [(p["seed"], p["iteration"]) for p in sel["policies"]] == [
        (0, 526), (0, 626), (0, 732), (1, 506), (1, 606), (1, 712), (2, 506), (2, 606), (2, 712)]


def test_author_lock_is_not_advisor_approval():
    assert LOCK["lock"]["advisor_approval"] == "not obtained"
    assert LOCK["lock"]["author_lock"]["locked_by"] == "project author"
    assert set(CFG.approval.values()) == {"pending"}


def test_evaluation_roles_and_confirmatory_levels():
    d = LOCK["densities"]
    assert d["primary"] == [1, 2, 3] and d["diagnostic"] == [4] and d["stress"] == [5, 8]
    c = LOCK["confirmatory"]
    assert abs(c["alpha_density"] - 0.05 / 3) < 1e-15
    assert c["goal_loss_margin"] == 0.05 and c["formation_margin_m"] == 0.01
    assert tuple(LOCK["gains"]["grid"]) == CFG.alphas
    assert (CFG.protocol_config().goal_tol, CFG.protocol_config().formation_tol) == (0.10, 0.10)


def test_zero_command_separation_prediction():
    got = {}
    for a in (0.5, 1.0):
        fc = CFG.filter_config(a)
        got[a] = brentq(lambda d: a * (d * d - fc.d_rr ** 2) - margin_robot(fc, d), fc.d_rr, 10.0)
    assert abs(got[0.5] - 0.668) < 1e-3 and abs(got[1.0] - 0.501) < 1e-3
    spacing = CFG["scenario.slot_radius"] * np.sqrt(3)
    assert abs(spacing - 0.520) < 1e-3
    assert got[0.5] > spacing > got[1.0]
