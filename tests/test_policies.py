"""The frozen policy is deterministic, gradient-free and unchanged by an evaluation run, and is
refused under an environment it was not trained in."""
import tomllib
from dataclasses import replace

import numpy as np
import pytest
import torch

from cbf_study.config import DEFAULT_PATH, StudyConfig
from cbf_study.envs import KinematicEnv
from cbf_study.mappo import Actor, save_actor
from cbf_study.observations import obs_dim
from cbf_study.policies import FrozenMAPPOPolicy
from cbf_study.protocol import run_paired, run_trials
from cbf_study.scenario import sample_layout

CFG = StudyConfig.load()


def random_checkpoint(path, cfg=CFG, seed=0):
    """An untrained actor saved in the training checkpoint format (no trained artefact needed)."""
    torch.manual_seed(seed)
    fc = cfg.layout_filter_config()
    meta = dict(obs_dim=obs_dim(cfg.obs_config()), n_agents=cfg.obs_config().n_agents, hidden_actor=[32, 32],
                lo=[fc.v_min, -fc.w_max], hi=[fc.v_max, fc.w_max], init_std_frac=0.5,
                env_fingerprint=cfg.env_fingerprint(), config=cfg.as_dict(), iteration=0, env_steps=0)
    actor = Actor(meta["obs_dim"], meta["hidden_actor"], meta["lo"], meta["hi"], 0.5)
    with torch.no_grad():                       # make the mean non-trivial
        for p in actor.body.parameters():
            p.mul_(30.0)
    save_actor(path, actor, meta)
    return path


def test_two_calls_give_identical_actions_inside_the_box(tmp_path):
    pol = FrozenMAPPOPolicy(random_checkpoint(tmp_path / "c.pt"))
    lay = sample_layout(3, 16, 5, CFG.layout_filter_config(), CFG.scenario_config())
    a1 = pol(lay.init_poses, lay)
    a2 = pol(lay.init_poses.copy(), lay)
    assert a1.shape == (16, 3, 2) and a1.dtype == np.float64
    assert np.array_equal(a1, a2)
    assert (a1 >= pol.lo).all() and (a1 <= pol.hi).all() and np.abs(a1).max() > 0.05
    # a second, independently loaded copy gives the same actions
    assert np.array_equal(FrozenMAPPOPolicy(tmp_path / "c.pt")(lay.init_poses, lay), a1)


def test_no_parameter_changes_during_an_evaluation_run(tmp_path):
    pol = FrozenMAPPOPolicy(random_checkpoint(tmp_path / "c.pt"))
    assert not pol.actor.training
    assert not any(p.requires_grad for p in pol.actor.parameters())
    before = {k: v.clone() for k, v in pol.actor.state_dict().items()}
    fc, pc = CFG.filter_config(2.0), replace(CFG.protocol_config(), max_steps=60)
    lay = sample_layout(4, 8, 5, CFG.layout_filter_config(), CFG.scenario_config())
    off, on = run_paired(lambda: KinematicEnv(pc.dt), pol, lay, pc, fc, log_steps=False)
    pol.assert_unchanged()
    after = pol.actor.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    # and the evaluation itself is reproducible
    again = run_trials(KinematicEnv(pc.dt), pol, lay, pc, fc, use_filter=True, log_steps=False)
    assert np.array_equal(again.outcome, on.outcome) and np.array_equal(again.final_poses, on.final_poses)


def test_checkpoint_refused_under_a_different_environment(tmp_path):
    ck = random_checkpoint(tmp_path / "c.pt")
    with open(DEFAULT_PATH, "rb") as f:
        raw = tomllib.load(f)
    raw["robot"]["v_max"]["value"] = 0.4
    with pytest.raises(ValueError, match="fingerprint"):
        FrozenMAPPOPolicy(ck, StudyConfig(raw))
    with open(DEFAULT_PATH, "rb") as f:
        raw = tomllib.load(f)
    raw["filter"]["alpha_sweep"]["value"] = [1.0]          # filter-only change is fine
    FrozenMAPPOPolicy(ck, StudyConfig(raw))
