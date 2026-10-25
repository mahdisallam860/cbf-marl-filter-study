"""
Adapter for a stock VMAS scenario (default: "navigation"), used only to validate the MAPPO
implementation on a task it was not written for. Same interface as FormationTrainEnv:
reset() -> obs [E,N,D]; step(raw [E,N,A]) -> StepOut; n_envs, lo, hi.

* actions: the raw Gaussian sample is clipped to the agent's u_range box before VMAS sees it
  (last_applied records what was applied);
* reward: the scenario's shared team reward (shared_rew=True), agent 0's copy;
* terminated: the scenario's done() (navigation: every agent on its goal);
  truncated: VMAS's max_steps time limit, and only when not terminated;
* auto-reset: finished envs are reset with reset_at; final_obs is the observation before reset.
"""
from __future__ import annotations

import numpy as np
import torch

from .train_env import StepOut


class VmasStockEnv:
    def __init__(self, scenario="navigation", n_envs=64, n_agents=3, max_steps=100, seed=0, **kwargs):
        from vmas import make_env
        self.env = make_env(scenario=scenario, num_envs=n_envs, device="cpu", continuous_actions=True, seed=seed,
                            max_steps=max_steps, terminated_truncated=True, n_agents=n_agents, **kwargs)
        self.n_envs, self.n_agents, self.max_steps = n_envs, n_agents, max_steps
        a0 = self.env.agents[0]
        A = self.env.get_agent_action_size(a0)
        u = torch.as_tensor(a0.u_range, dtype=torch.float32).expand(A).numpy().astype(np.float64)
        self.lo, self.hi = -u, u
        self.ret = np.zeros(n_envs)
        self.len = np.zeros(n_envs, dtype=np.int64)
        self.last_applied = None

    @staticmethod
    def _stack(obs):
        return torch.stack(obs, dim=1).numpy().astype(np.float32)

    def reset(self):
        self.ret[:] = 0; self.len[:] = 0
        return self._stack(self.env.reset())

    def step(self, raw):
        a = np.clip(np.asarray(raw, dtype=np.float64), self.lo, self.hi)
        self.last_applied = a
        obs, rews, term, trunc, _ = self.env.step([torch.as_tensor(a[:, i], dtype=torch.float32)
                                                   for i in range(self.n_agents)])
        r = rews[0].numpy().astype(np.float64)
        term = term.numpy().astype(bool)
        trunc = trunc.numpy().astype(bool) & ~term
        final_obs = self._stack(obs)
        self.ret += r; self.len += 1
        done = term | trunc
        episodes = [dict(ret=float(self.ret[i]), length=int(self.len[i]), success=bool(term[i]))
                    for i in np.flatnonzero(done)]
        for i in np.flatnonzero(done):
            self.env.reset_at(int(i), return_observations=False)
        self.ret[done] = 0; self.len[done] = 0
        new_obs = self._stack(self.env.get_from_scenario(get_observations=True, get_rewards=False,
                                                         get_infos=False, get_dones=False)[0]) if done.any() else final_obs
        return StepOut(new_obs, r, term, trunc, final_obs, episodes)
