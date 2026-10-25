"""MAPPO implementation checks: bounded applied actions, consistent log-probabilities, PPO ratio
exactly 1 before the first gradient step, termination vs time-limit bootstrapping, and actors that
see only their own observation. Run on the formation env, a stock VMAS scenario and a synthetic env."""
import numpy as np
import pytest
import torch

from cbf_study.config import StudyConfig
from cbf_study.mappo import MappoLearner
from cbf_study.observations import build_obs, obs_dim
from cbf_study.train_env import StepOut

pytest.importorskip("vmas")
from cbf_study.stock_env import VmasStockEnv          # noqa: E402
from cbf_study.train_env import FormationTrainEnv     # noqa: E402

CFG = StudyConfig.load()
HP = dict(CFG.mappo, epochs=2, minibatches=2, hidden_actor=[32, 32], hidden_critic=[32, 32])


def formation_env(n=4):
    env = FormationTrainEnv(CFG, run_seed=0, envs_per_density=n)
    applied = []
    for g in env.groups:                               # record what VMAS actually receives
        orig = g.env.step
        g.env.step = (lambda u, orig=orig: (applied.append(np.array(u)), orig(u))[1])
    return env, applied


@pytest.mark.parametrize("which", ["formation", "vmas_navigation"])
def test_applied_actions_bounded_logp_consistent_ratio_one(which):
    if which == "formation":
        env, applied = formation_env()
        D, N = obs_dim(CFG.obs_config()), 3
    else:
        env = VmasStockEnv(n_envs=8, n_agents=3, max_steps=20, seed=0)
        applied = []
        orig = env.step
        env.step = lambda raw: (lambda so: (applied.append(env.last_applied), so)[1])(orig(raw))
        N = 3
        D = env.reset().shape[2]
    lrn = MappoLearner(D, N, env.lo, env.hi, dict(HP, init_std_frac=2.0), seed=0)   # wide std: samples leave the box
    obs = env.reset()
    b, _, _ = lrn.collect(env, obs, 12)
    raw = b["act"]
    assert (raw < env.lo).any() or (raw > env.hi).any()            # the raw samples are unbounded ...
    app = np.concatenate(applied)
    assert (app >= env.lo - 1e-6).all() and (app <= env.hi + 1e-6).all()   # ... what is applied is clipped
    with torch.no_grad():
        lp = lrn.actor.log_prob(torch.from_numpy(b["obs"]), torch.from_numpy(raw)).numpy()
    assert np.allclose(lp, b["logp"], atol=1e-4)                    # log-prob of the raw sample, same as in update
    st = lrn.update(b)
    assert st["ratio0_maxdev"] < 1e-5                               # ratio == 1 before any gradient step
    m = lrn.actor.mean(torch.randn(500, D) * 50).detach().numpy()   # deterministic action always in the box
    assert (m >= env.lo - 1e-6).all() and (m <= env.hi + 1e-6).all()


class SyntheticEnv:
    """reward 1 every step; env 0 is truncated every 3 steps, env 1 terminates every 4 steps."""
    def __init__(self):
        self.n_envs, self.t = 2, np.zeros(2, int)
        self.lo, self.hi = np.array([-1.0]), np.array([1.0])

    def reset(self):
        return np.zeros((2, 1, 1), np.float32)

    def step(self, raw):
        self.t += 1
        trunc = np.array([self.t[0] % 3 == 0, False])
        term = np.array([False, self.t[1] % 4 == 0])
        final = np.full((2, 1, 1), 7.0, np.float32)                  # pre-reset observation
        return StepOut(np.zeros((2, 1, 1), np.float32), np.ones(2), term, trunc, final, [])


def test_time_limit_is_bootstrapped_and_termination_is_not():
    lrn = MappoLearner(1, 1, [-1.0], [1.0], HP, seed=0)
    c, gamma = 5.0, float(HP["gamma"])
    seen = []
    lrn.value = lambda o: (seen.append(o.copy()), np.full(o.shape[0], c))[1]
    env = SyntheticEnv()
    b, _, _ = lrn.collect(env, env.reset(), 12)
    t = np.arange(1, 13)
    exp_rew0 = np.where(t % 3 == 0, 1.0 + gamma * c, 1.0)           # truncation: bootstrap from V(final_obs)
    assert np.allclose(b["rew"][:, 0], exp_rew0)
    assert np.allclose(b["rew"][:, 1], 1.0)                          # termination: no bootstrap
    assert np.array_equal(b["done"][:, 0], (t % 3 == 0).astype(float))
    assert np.array_equal(b["done"][:, 1], (t % 4 == 0).astype(float))
    assert any((o == 7.0).all() for o in seen)                       # the bootstrap used the pre-reset state


def test_actor_sees_only_its_own_observation_row():
    env = FormationTrainEnv(CFG, run_seed=0, envs_per_density=2)
    obs = env.reset()
    # the training observation is exactly build_obs(poses, layout), nothing else
    assert np.array_equal(obs, np.concatenate([build_obs(g.poses, g.layout, CFG.obs_config()) for g in env.groups]))
    D = obs.shape[2]
    lrn = MappoLearner(D, 3, env.lo, env.hi, HP, seed=0)
    with torch.no_grad():
        o = torch.from_numpy(obs)
        m0 = lrn.actor.mean(o)
        o2 = o.clone()
        o2[:, 1:] += torch.randn_like(o2[:, 1:]) * 10               # change robots 1 and 2 only
        m1 = lrn.actor.mean(o2)
        assert torch.equal(m0[:, 0], m1[:, 0])                      # robot 0's action is unchanged
        assert not torch.equal(m0[:, 1], m1[:, 1])
        v0 = lrn.critic(o.reshape(len(o), -1)); v1 = lrn.critic(o2.reshape(len(o), -1))
        assert not torch.equal(v0, v1)                              # only the centralised critic sees all rows
