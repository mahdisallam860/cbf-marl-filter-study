"""The study config is the single source of placeholders, every entry carries an approval state,
and filter-on / filter-off trials are guaranteed to use identical values."""
import tomllib
from dataclasses import fields, replace

import numpy as np
import pytest

from cbf_study.cbf_qp import FilterConfig
from cbf_study.config import DEFAULT_PATH, ScenarioConfig, StudyConfig, check_paired
from cbf_study.protocol import ProtocolConfig
from cbf_study.scenario import sample_layout


def raw():
    with open(DEFAULT_PATH, "rb") as f:
        return tomllib.load(f)


def test_every_study_value_has_an_approval_state_and_all_are_pending():
    cfg = StudyConfig.load()
    n = sum(len(v) for v in cfg.values.values())
    assert len(cfg.approval) == n > 40
    # nothing has been approved by an advisor yet
    assert cfg.pending() == sorted(cfg.approval)


def test_entry_without_approval_is_rejected():
    r = raw()
    r["robot"]["v_max"] = 0.5
    with pytest.raises(ValueError, match="approval"):
        StudyConfig(r)
    r = raw()
    r["robot"]["v_max"]["approval"] = "ok"
    with pytest.raises(ValueError, match="approval"):
        StudyConfig(r)


def test_dataclasses_are_fully_set_from_the_file():
    """No study path may rely on a dataclass default: every field of FilterConfig, ProtocolConfig
    and ScenarioConfig must come from a key in the file."""
    cfg = StudyConfig.load()
    keys = {k.split(".")[1] for k in cfg.approval}
    derived = {"dt", "alpha"}               # dt from [timing], alpha from the sweep
    for cls in (FilterConfig, ProtocolConfig, ScenarioConfig):
        missing = {f.name for f in fields(cls)} - keys - derived
        assert not missing, (cls.__name__, missing)
    fc, pc, sc = cfg.filter_config(2.0), cfg.protocol_config(), cfg.scenario_config()
    assert fc.ell == cfg["filter.ell"] and pc.goal_tol == cfg["protocol.goal_tol"]
    assert sc.start_xy_std == cfg["scenario.start_xy_std"] and sc.obst_y == tuple(cfg["scenario.obst_y"])


def test_alpha_sweep_respects_lemma_hypothesis():
    cfg = StudyConfig.load()
    assert all(a * cfg.dt <= 1.0 + 1e-12 for a in cfg.alphas)
    r = raw()
    r["filter"]["alpha_sweep"]["value"] = [2.0, 20.0]
    with pytest.raises(ValueError, match="alpha"):
        StudyConfig(r)


def test_filter_on_and_off_use_identical_values():
    cfg = StudyConfig.load()
    pc = cfg.protocol_config()
    fc_off = cfg.filter_config(cfg.alphas[0])
    for a in cfg.alphas:
        check_paired(pc, fc_off, cfg.protocol_config(), cfg.filter_config(a))
    with pytest.raises(ValueError):
        check_paired(pc, fc_off, replace(pc, v_max=0.4), cfg.filter_config(2.0))
    with pytest.raises(ValueError):
        check_paired(pc, fc_off, pc, replace(cfg.filter_config(2.0), ell=0.06))
    with pytest.raises(ValueError):
        check_paired(pc, fc_off, pc, replace(cfg.filter_config(2.0), sensing_range=1.2))


def test_default_config_reproduces_the_existing_layouts():
    """The file holds the same placeholder values the harness used before, so the layout sampler
    gives bit-identical layouts with the config-built objects and with the old defaults."""
    cfg = StudyConfig.load()
    a = sample_layout(7, 5, 5, cfg.layout_filter_config(), cfg.scenario_config())
    b = sample_layout(7, 5, 5, cfg.layout_filter_config())
    assert np.array_equal(a.init_poses, b.init_poses) and np.array_equal(a.obst_xy, b.obst_xy)


def test_fingerprint_tracks_environment_values_only():
    cfg = StudyConfig.load()
    f0 = cfg.env_fingerprint()
    r = raw()
    r["filter"]["alpha_sweep"]["value"] = [1.0]          # filter-only value: same environment
    r["reward"]["w_progress"]["value"] = 3.0             # training-only value: same environment
    assert StudyConfig(r).env_fingerprint() == f0
    for sec, key, val in (("robot", "v_max", 0.4), ("filter", "ell", 0.06), ("protocol", "goal_tol", 0.2),
                          ("observation", "k_obst", 3), ("scenario", "obst_gap", 0.2)):
        r = raw()
        r[sec][key]["value"] = val
        assert StudyConfig(r).env_fingerprint() != f0, (sec, key)


def test_seed_ranges_are_disjoint():
    cfg = StudyConfig.load()
    train = {cfg.train_seed(s, k, c) for s in range(10) for k in cfg.densities for c in (0, 999)}
    val = {cfg.validation_seed(k) for k in cfg.densities}
    pil = {cfg.pilot_seed(k) for k in cfg.densities}
    assert not (train & val) and not (train & pil) and not (val & pil)
    assert max(train) < min(val) < max(val) < min(pil)
