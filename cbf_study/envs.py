"""
Batched environments with one interface, so every protocol rule is shared.

    env.reset(init_poses [B,N,3], obst_xy [B,K,2])
    env.poses()          -> [B,N,3] float64 axle-centre poses
    env.step(u [B,N,2])  -> advances one control period with command (v, w) held constant

VmasEnv    : the pinned VMAS 1.5.2 DiffDrive scenario, configured exactly as the gate test
             recommends: world drag = 0, RK4, collide=False (no engine contacts). Collisions are
             never taken from the engine; they are scored geometrically in protocol.py.
KinematicEnv: closed-form unicycle in float64. Used as an independent reference and for fast tests.
"""
from __future__ import annotations

import numpy as np

from .geometry import arc_poses


class KinematicEnv:
    def __init__(self, dt):
        self.dt = dt
        self._poses = None
        self.obst_xy = None

    def reset(self, init_poses, obst_xy):
        self._poses = np.array(init_poses, dtype=np.float64)
        self.obst_xy = np.array(obst_xy, dtype=np.float64)

    def poses(self):
        return self._poses.copy()

    def step(self, u):
        B, N, _ = self._poses.shape
        flat = self._poses.reshape(B * N, 3)
        uu = np.asarray(u, dtype=np.float64).reshape(B * N, 2)
        end = arc_poses(flat, uu[:, 0], uu[:, 1], self.dt, 1)[:, -1, :]
        self._poses = end.reshape(B, N, 3)


class VmasEnv:
    """VMAS 1.5.2 DiffDrive, drag 0, RK4, no engine contacts."""

    def __init__(self, batch, n_agents, n_obst, dt, robot_r, obst_r, v_max, w_max, seed=0):
        import torch
        from vmas import make_env
        from vmas.simulator.core import Agent, Landmark, Sphere, World
        from vmas.simulator.dynamics.diff_drive import DiffDrive
        from vmas.simulator.scenario import BaseScenario

        self._torch = torch
        self.dt = dt
        self.v_max, self.w_max = v_max, w_max
        outer = self

        class _Scenario(BaseScenario):
            def make_world(self, batch_dim, device, **kw):
                world = World(batch_dim, device, dt=dt, drag=0.0)
                for i in range(n_agents):
                    world.add_agent(Agent(name=f"robot_{i}", shape=Sphere(robot_r), collide=False,
                                          movable=True, rotatable=True, u_range=[v_max, w_max],
                                          dynamics=DiffDrive(world, integration="rk4")))
                for j in range(n_obst):
                    world.add_landmark(Landmark(name=f"obst_{j}", shape=Sphere(obst_r),
                                                movable=False, collide=False))
                return world

            def reset_world_at(self, env_index=None):
                for i, a in enumerate(self.world.agents):
                    a.set_pos(torch.tensor([10.0 * i, 10.0]), batch_index=env_index)
                for j, l in enumerate(self.world.landmarks):
                    l.set_pos(torch.tensor([10.0 * j, -10.0]), batch_index=env_index)

            def observation(self, agent):
                return torch.zeros(self.world.batch_dim, 1, device=self.world.device)

            def reward(self, agent):
                return torch.zeros(self.world.batch_dim, device=self.world.device)

        self.env = make_env(scenario=_Scenario(), num_envs=batch, device="cpu",
                            continuous_actions=True, seed=seed)
        self.env.reset()
        self.n_agents, self.n_obst, self.batch = n_agents, n_obst, batch

    def reset(self, init_poses, obst_xy):
        torch = self._torch
        init_poses = np.asarray(init_poses)
        for i, a in enumerate(self.env.agents):
            p = torch.as_tensor(init_poses[:, i, :], dtype=torch.float32)
            a.set_pos(p[:, :2], batch_index=None)
            a.set_rot(p[:, 2:3], batch_index=None)
            a.set_vel(torch.zeros_like(p[:, :2]), batch_index=None)
            a.set_ang_vel(torch.zeros_like(p[:, 2:3]), batch_index=None)
        obst_xy = np.asarray(obst_xy)
        for j, l in enumerate(self.env.world.landmarks):
            l.set_pos(torch.as_tensor(obst_xy[:, j, :], dtype=torch.float32), batch_index=None)

    def poses(self):
        out = [torch_cat(self._torch, a) for a in self.env.agents]
        return np.stack(out, axis=1)

    def step(self, u):
        torch = self._torch
        u = np.asarray(u, dtype=np.float64)
        # VMAS does not clip DiffDrive commands in a way we rely on: the protocol clips first.
        assert np.all(np.abs(u[..., 0]) <= self.v_max + 1e-6) and np.all(np.abs(u[..., 1]) <= self.w_max + 1e-6)
        acts = [torch.as_tensor(u[:, i, :], dtype=torch.float32) for i in range(self.n_agents)]
        self.env.step(acts)


def torch_cat(torch, agent):
    return torch.cat([agent.state.pos, agent.state.rot], dim=1).double().numpy()
