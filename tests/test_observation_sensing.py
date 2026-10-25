"""Config v2 checks: every robot observes every obstacle within the declared observation range, in its
own body frame, and nothing beyond it; the proximity shaping term; v1 stays reproducible."""
from pathlib import Path

import numpy as np
import pytest

from cbf_study.config import StudyConfig
from cbf_study.evaluate import eval_layout
from cbf_study.observations import build_obs
from cbf_study.scenario import Layout
from cbf_study.train_env import proximity_penalty

CFG = StudyConfig.load()
ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "config" / "archive" / "study_v1.toml"


def obstacle_block(obs, oc):
    N = oc.n_agents
    start = 4 + 4 * (N - 1)
    return obs[..., start:start + 4 * oc.k_obst].reshape(obs.shape[:2] + (oc.k_obst, 4))


@pytest.mark.parametrize("K", [0, 1, 2, 3, 5, 8])
def test_every_obstacle_within_range_is_observed_in_the_body_frame(K):
    oc = CFG.obs_config()
    assert CFG.observes_all_obstacles
    lay = eval_layout(CFG, K, "validation", 40)
    rng = np.random.default_rng(K)
    for s in (0.0, 0.3, 0.6, 1.0):                      # start, inside the field, at the goal
        P = lay.init_poses.copy()
        cen = (1 - s) * P[..., :2].mean(1) + s * lay.goal
        P[..., :2] = cen[:, None] + lay.slots[None] + rng.normal(0, 0.05, P[..., :2].shape)
        P[..., 2] = rng.uniform(-np.pi, np.pi, P[..., 2].shape)
        blk = obstacle_block(build_obs(P, lay, oc), oc).astype(np.float64)
        for b in range(len(P)):
            for i in range(3):
                rel = lay.obst_xy[b] - P[b, i, :2]                       # world frame
                d = np.linalg.norm(rel, axis=1)
                inr = np.sort(d[d <= oc.obs_range])
                present = blk[b, i, :, 3] > 0.5
                assert present.sum() == len(inr)                          # every obstacle in range, none beyond
                th = P[b, i, 2]
                Rt = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])   # world -> body
                body = (Rt @ rel[d <= oc.obs_range].T).T
                got = blk[b, i, present, :2]
                order = np.argsort(np.linalg.norm(body, axis=1), kind="stable")
                assert np.allclose(got, body[order], atol=1e-5)          # body-frame coordinates, nearest first
                assert np.allclose(blk[b, i, present, 2], inr - (oc.robot_r + oc.obst_r), atol=1e-5)
                assert (blk[b, i, ~present] == np.array([0, 0, oc.obs_range, 0])).all()


def test_v1_observation_dropped_obstacles_in_range():
    """the reason for v2: with k_obst = 4, robots at K = 8 had more obstacles in range than observed."""
    v1 = StudyConfig.load(V1)
    assert not v1.observes_all_obstacles and v1.env_fingerprint() == "cf442ee5e99eb55a"
    lay = eval_layout(v1, 8, "validation", 10)
    d = np.linalg.norm(lay.obst_xy, axis=2)                        # a robot at the centre of the field
    assert ((d <= v1.obs_config().obs_range).sum(axis=1) > v1.obs_config().k_obst).all()


def test_v1_checkpoints_load_only_under_v1():
    from cbf_study.policies import FrozenMAPPOPolicy
    ck = ROOT / "results" / "train" / "seed_0" / "checkpoints" / "ckpt_it0300.pt"
    if not ck.exists():
        pytest.skip("v1 checkpoints not present")
    FrozenMAPPOPolicy(ck, StudyConfig.load(V1))
    with pytest.raises(ValueError, match="fingerprint"):
        FrozenMAPPOPolicy(ck, CFG)


def test_proximity_penalty():
    pc, rc = CFG.protocol_config(), CFG.reward_config()
    assert rc.w_proximity > 0 and rc.proximity_margin < np.sqrt(3) * 0.30 - 2 * pc.robot_r   # none in formation
    slots = CFG.scenario_config()
    lay = eval_layout(CFG, 2, "validation", 4)
    assert (proximity_penalty(lay.init_poses, np.zeros((4, 0, 2)), pc, rc.proximity_margin) == 0).all()
    P = np.zeros((1, 3, 3)); P[0, :, 0] = [0.0, 5.0, 10.0]
    m = rc.proximity_margin
    for clr, exp in ((0.2, 0.0), (m, 0.0), (m / 2, 0.5), (0.0, 1.0)):
        O = np.array([[[pc.robot_r + pc.obst_r + clr, 0.0]]])
        assert np.isclose(proximity_penalty(P, O, pc, m)[0], exp)


def test_curriculum_is_fixed_in_the_config():
    cur = CFG.curriculum()
    assert cur["stages"][0] == (0,) and tuple(cur["stages"][-1]) == CFG.densities
    assert (2,) in cur["stages"] and len(cur["budget"]) == len(cur["stages"])
    assert cur["k2_stop_goal_rate"] == 0.5
