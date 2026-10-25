"""MAPPO smoke test: a tiny run trains, logs curves and writes loadable actor checkpoints."""
import csv

import numpy as np
import pytest
import torch

pytest.importorskip("vmas")
from cbf_study.config import StudyConfig            # noqa: E402
from cbf_study.mappo import _gae, load_actor_checkpoint, train  # noqa: E402

TINY = dict(envs_per_density=4, rollout_steps=16, max_iterations=2, checkpoint_every=1, minibatches=2, epochs=2)


def test_gae_matches_hand_computation():
    rew = np.array([[1.0], [0.0], [2.0]]); val = np.array([[0.5], [0.2], [0.1]])
    done = np.array([[0.0], [1.0], [0.0]]); last = np.array([0.3])
    g, l = 0.9, 0.8
    d2 = 2.0 + g * 0.3 - 0.1
    d1 = 0.0 - 0.2                                     # episode ends at t=1: no bootstrap
    d0 = 1.0 + g * 0.2 - 0.5
    exp = np.array([[d0 + g * l * d1], [d1], [d2]])
    assert np.allclose(_gae(rew, val, done, last, g, l), exp)


def test_tiny_training_run_writes_curves_and_checkpoints(tmp_path):
    cfg = StudyConfig.load()
    train(cfg, 0, tmp_path, overrides=TINY, log=lambda m: None)
    rows = list(csv.DictReader(open(tmp_path / "curve.csv")))
    assert len(rows) == 2 and int(rows[-1]["env_steps"]) == 2 * 16 * 4 * len(cfg.densities)
    ck = sorted((tmp_path / "checkpoints").glob("ckpt_it*.pt"))
    assert [p.name for p in ck] == ["ckpt_it0001.pt", "ckpt_it0002.pt"]
    actor, meta = load_actor_checkpoint(ck[-1])
    assert meta["env_fingerprint"] == cfg.env_fingerprint() and meta["iteration"] == 2
    # the mean action lies in the velocity box for any input
    m = actor.mean(torch.randn(1000, meta["obs_dim"]) * 100).detach().numpy()
    assert (m >= np.array(meta["lo"]) - 1e-6).all() and (m <= np.array(meta["hi"]) + 1e-6).all()
    # resume continues from the saved iteration
    train(cfg, 0, tmp_path, overrides=dict(TINY, max_iterations=3), resume=True, log=lambda m: None)
    assert len(list(csv.DictReader(open(tmp_path / "curve.csv")))) == 3
