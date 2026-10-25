#!/usr/bin/env python3
"""
Simulator-selection gate test: VMAS DiffDrive vs an independent unicycle model.

Purpose
-------
Decide whether VMAS (pinned) executes a commanded (forward, angular) velocity as an
ideal kinematic unicycle, so that a CBF filter written for the unicycle model is tested
against a simulator that obeys the same model. No MAPPO training, no CBF, no safety claim.

Part A  (free flight, no contacts)
    For every VMAS step, take VMAS's own pose before the step, apply the commanded
    (v, w) with an independently implemented closed-form unicycle step (float64,
    exact arc), and compare with the pose VMAS actually reaches.
    Logged per step: commanded action, realized pose change, reference pose change,
    one-step position and heading error.

Part B  (robot-obstacle, robot-robot, close approach)
    Geometric collision events are scored from trajectories only (center distance vs
    sum of radii), independent of VMAS's contact response. Each case is run three ways:
      kin     : independent open-loop unicycle reference (no VMAS)
      vmas_off: VMAS with contact response disabled (collide=False)
      vmas_on : VMAS with contact response enabled (collide=True)
    Contact-induced departure = |pose(vmas_on) - pose(vmas_off)|.
    Model/drag departure     = |pose(vmas_off) - pose(kin)|.

Usage:  python gate_test_vmas_diffdrive.py --out results
"""
import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import torch
import vmas
from vmas import make_env
from vmas.simulator.core import Agent, Landmark, Sphere, World
from vmas.simulator.dynamics.diff_drive import DiffDrive
from vmas.simulator.scenario import BaseScenario

SEED = 20261025
V_MAX = 0.5      # m/s   placeholder command bound (not taken from the study)
W_MAX = 1.0      # rad/s placeholder command bound
ROBOT_R = 0.10   # m     placeholder robot radius
OBST_R = 0.20    # m     placeholder obstacle radius


# --------------------------------------------------------------------------------------
# VMAS scenario used only as a test fixture
# --------------------------------------------------------------------------------------
class GateScenario(BaseScenario):
    def make_world(self, batch_dim, device, **kwargs):
        self.n_agents = kwargs.get("n_agents", 1)
        self.n_obst = kwargs.get("n_obst", 0)
        self.contact = kwargs.get("contact", False)
        world = World(
            batch_dim,
            device,
            dt=kwargs.get("dt", 0.1),
            drag=kwargs.get("drag", 0.25),  # VMAS default is 0.25
        )
        for i in range(self.n_agents):
            world.add_agent(
                Agent(
                    name=f"robot_{i}",
                    shape=Sphere(ROBOT_R),
                    collide=self.contact,
                    movable=True,
                    rotatable=True,
                    u_range=[V_MAX, W_MAX],
                    dynamics=DiffDrive(world, integration=kwargs.get("integration", "rk4")),
                )
            )
        for j in range(self.n_obst):
            world.add_landmark(
                Landmark(name=f"obst_{j}", shape=Sphere(OBST_R), movable=False, collide=self.contact)
            )
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


def make_fixture(batch, n_agents, n_obst, contact, drag, integration, dt):
    env = make_env(
        scenario=GateScenario(),
        num_envs=batch,
        device="cpu",
        continuous_actions=True,
        seed=SEED,
        n_agents=n_agents,
        n_obst=n_obst,
        contact=contact,
        drag=drag,
        integration=integration,
        dt=dt,
    )
    env.reset()
    return env


def set_state(env, poses_agents, poses_obst=None):
    """poses_agents: list of arrays [B,3] (x, y, theta). Velocities start at zero."""
    for a, p in zip(env.agents, poses_agents):
        p = torch.as_tensor(p, dtype=torch.float32)
        a.set_pos(p[:, :2], batch_index=None)
        a.set_rot(p[:, 2:3], batch_index=None)
        a.set_vel(torch.zeros_like(p[:, :2]), batch_index=None)
        a.set_ang_vel(torch.zeros_like(p[:, 2:3]), batch_index=None)
    if poses_obst is not None:
        for l, p in zip(env.world.landmarks, poses_obst):
            p = torch.as_tensor(p, dtype=torch.float32)
            l.set_pos(p[:, :2], batch_index=None)


def get_pose(agent):
    return torch.cat([agent.state.pos, agent.state.rot], dim=1).double().numpy()  # [B,3]


# --------------------------------------------------------------------------------------
# Independent unicycle reference (closed-form arc, float64) - does not call VMAS
# --------------------------------------------------------------------------------------
def unicycle_step(pose, v, w, dt):
    x, y, th = pose[:, 0], pose[:, 1], pose[:, 2]
    wdt = w * dt
    small = np.abs(w) < 1e-9
    safe_w = np.where(small, 1.0, w)
    dx = np.where(small, v * dt * np.cos(th), (v / safe_w) * (np.sin(th + wdt) - np.sin(th)))
    dy = np.where(small, v * dt * np.sin(th), (v / safe_w) * (-np.cos(th + wdt) + np.cos(th)))
    return np.stack([x + dx, y + dy, th + wdt], axis=1)


# --------------------------------------------------------------------------------------
# Part A: free flight
# --------------------------------------------------------------------------------------
def part_a(out_dir, batch=100, n_agents=3, steps=20):
    rows, summary = [], []
    cfg_id = 0
    for dt in (0.1, 0.05):
        for drag in (0.25, 0.0):
            for integ in ("rk4", "euler"):
                for mode in ("const", "rand"):
                    cfg_id += 1
                    rng = np.random.default_rng(SEED + cfg_id)
                    env = make_fixture(batch, n_agents, 0, False, drag, integ, dt)
                    init = [
                        np.stack(
                            [rng.uniform(-1, 1, batch), rng.uniform(-1, 1, batch),
                             rng.uniform(-math.pi, math.pi, batch)], axis=1)
                        for _ in range(n_agents)
                    ]
                    set_state(env, init)
                    if mode == "const":
                        cv = [rng.uniform(0.0, V_MAX, batch) for _ in range(n_agents)]
                        cw = [rng.uniform(-W_MAX, W_MAX, batch) for _ in range(n_agents)]
                    pos_err_all, head_err_all = [], []
                    for k in range(steps):
                        if mode == "rand":
                            cv = [rng.uniform(0.0, V_MAX, batch) for _ in range(n_agents)]
                            cw = [rng.uniform(-W_MAX, W_MAX, batch) for _ in range(n_agents)]
                        before = [get_pose(a) for a in env.agents]
                        acts = [
                            torch.tensor(np.stack([cv[i], cw[i]], axis=1), dtype=torch.float32)
                            for i in range(n_agents)
                        ]
                        env.step(acts)
                        after = [get_pose(a) for a in env.agents]
                        for i in range(n_agents):
                            ref = unicycle_step(before[i], cv[i].astype(np.float32).astype(np.float64),
                                                cw[i].astype(np.float32).astype(np.float64), dt)
                            real_d = after[i] - before[i]
                            ref_d = ref - before[i]
                            perr = np.linalg.norm(after[i][:, :2] - ref[:, :2], axis=1)
                            herr = after[i][:, 2] - ref[:, 2]
                            pos_err_all.append((k, perr))
                            head_err_all.append((k, np.abs(herr)))
                            for b in range(batch):
                                rows.append([
                                    cfg_id, dt, drag, integ, mode, SEED + cfg_id, b, i, k,
                                    cv[i][b], cw[i][b],
                                    before[i][b, 0], before[i][b, 1], before[i][b, 2],
                                    real_d[b, 0], real_d[b, 1], real_d[b, 2],
                                    ref_d[b, 0], ref_d[b, 1], ref_d[b, 2],
                                    perr[b], herr[b],
                                ])
                    for scope, sel in (("step0", lambda k: k == 0), ("steps1+", lambda k: k >= 1),
                                       ("all", lambda k: True)):
                        pe = np.concatenate([p for k, p in pos_err_all if sel(k)])
                        he = np.concatenate([h for k, h in head_err_all if sel(k)])
                        summary.append([
                            cfg_id, dt, drag, integ, mode, scope, len(pe),
                            pe.mean(), np.percentile(pe, 50), np.percentile(pe, 95),
                            np.percentile(pe, 99), pe.max(),
                            he.mean(), np.percentile(he, 50), np.percentile(he, 95),
                            np.percentile(he, 99), he.max(),
                            pe.max() / (V_MAX * dt),
                        ])
    hdr = ["cfg", "dt", "drag", "integration", "cmd_mode", "seed", "env", "robot", "step",
           "cmd_v", "cmd_w", "x0", "y0", "th0", "real_dx", "real_dy", "real_dth",
           "ref_dx", "ref_dy", "ref_dth", "pos_err_m", "heading_err_rad"]
    with open(out_dir / "partA_free_flight_steps.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    shdr = ["cfg", "dt", "drag", "integration", "cmd_mode", "scope", "n",
            "pos_mean_m", "pos_p50_m", "pos_p95_m", "pos_p99_m", "pos_max_m",
            "head_mean_rad", "head_p50_rad", "head_p95_rad", "head_p99_rad", "head_max_rad",
            "pos_max_over_vmax_dt"]
    with open(out_dir / "partA_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(shdr); w.writerows(summary)
    return shdr, summary


# --------------------------------------------------------------------------------------
# Part B: geometric collision scoring and contact-induced departures
# --------------------------------------------------------------------------------------
def rollout_vmas(case, params, contact, drag, dt, steps):
    """Run one case in VMAS. Returns dict name -> [steps+1, B, 3] pose trajectories."""
    B = len(params)
    n_agents = 1 if case == "robot_obstacle" else 2
    n_obst = 1 if case == "robot_obstacle" else 0
    env = make_fixture(B, n_agents, n_obst, contact, drag, "rk4", dt)
    init, cmd, obst = case_setup(case, params)
    set_state(env, init, obst)
    traj = [[get_pose(a) for a in env.agents]]
    acts = [torch.tensor(c, dtype=torch.float32) for c in cmd]
    for _ in range(steps):
        env.step(acts)
        traj.append([get_pose(a) for a in env.agents])
    return [np.stack([t[i] for t in traj]) for i in range(n_agents)]  # per agent [T+1,B,3]


def rollout_kin(case, params, dt, steps):
    init, cmd, _ = case_setup(case, params)
    out = []
    for i, p0 in enumerate(init):
        p = np.asarray(p0, dtype=np.float64).copy()
        tr = [p.copy()]
        for _ in range(steps):
            p = unicycle_step(p, cmd[i][:, 0].astype(np.float64), cmd[i][:, 1].astype(np.float64), dt)
            tr.append(p.copy())
        out.append(np.stack(tr))
    return out


def case_setup(case, params):
    """params: lateral offsets [B]. Returns (initial poses, commands, obstacle poses)."""
    B = len(params)
    zeros = np.zeros(B)
    if case == "robot_obstacle":
        # robot heads +x at V_MAX toward an obstacle at the origin, laterally offset by params
        init = [np.stack([np.full(B, -1.0), params, zeros], axis=1)]
        cmd = [np.stack([np.full(B, V_MAX), zeros], axis=1)]
        obst = [np.stack([zeros, zeros, zeros], axis=1)]
    elif case == "robot_robot":
        # two robots approach head-on at V_MAX each, laterally offset by params
        init = [np.stack([np.full(B, -0.8), zeros, zeros], axis=1),
                np.stack([np.full(B, 0.8), params, np.full(B, math.pi)], axis=1)]
        cmd = [np.stack([np.full(B, V_MAX), zeros], axis=1),
               np.stack([np.full(B, V_MAX), zeros], axis=1)]
        obst = None
    else:
        raise ValueError(case)
    return init, cmd, obst


def separation(case, trajs, obst_xy=(0.0, 0.0)):
    """Center distance minus sum of radii, per step: [T+1,B]. Negative = geometric collision."""
    if case == "robot_obstacle":
        d = np.linalg.norm(trajs[0][:, :, :2] - np.array(obst_xy), axis=2)
        return d - (ROBOT_R + OBST_R)
    d = np.linalg.norm(trajs[0][:, :, :2] - trajs[1][:, :, :2], axis=2)
    return d - 2 * ROBOT_R


def first_event(sep):
    hit = sep < 0
    idx = np.where(hit.any(axis=0), hit.argmax(axis=0), -1)
    return idx


def part_b(out_dir, dt=0.1, steps=40):
    rows = []
    # lateral offsets chosen so the kinematic closest approach spans clear miss .. deep overlap
    offsets = {
        "robot_obstacle": np.linspace(0.0, OBST_R + ROBOT_R + 0.10, 31),
        "robot_robot": np.linspace(0.0, 2 * ROBOT_R + 0.10, 31),
    }
    for case in ("robot_obstacle", "robot_robot"):
        for drag in (0.0, 0.25):
            params = offsets[case]
            kin = rollout_kin(case, params, dt, steps)
            off = rollout_vmas(case, params, False, drag, dt, steps)
            on = rollout_vmas(case, params, True, drag, dt, steps)
            s_kin, s_off, s_on = separation(case, kin), separation(case, off), separation(case, on)
            e_kin, e_off, e_on = first_event(s_kin), first_event(s_off), first_event(s_on)
            for b, lat in enumerate(params):
                dep_on_off = np.max(np.stack(
                    [np.linalg.norm(on[i][:, b, :2] - off[i][:, b, :2], axis=1) for i in range(len(on))]), axis=0)
                dep_off_kin = np.max(np.stack(
                    [np.linalg.norm(off[i][:, b, :2] - kin[i][:, b, :2], axis=1) for i in range(len(on))]), axis=0)
                first_dep = int(np.argmax(dep_on_off > 1e-3)) if (dep_on_off > 1e-3).any() else -1
                rows.append([
                    case, drag, dt, steps, SEED, b, round(float(lat), 5),
                    float(s_kin[:, b].min()), float(s_off[:, b].min()), float(s_on[:, b].min()),
                    int(e_kin[b]), int(e_off[b]), int(e_on[b]),
                    float(np.maximum(0.0, -s_on[:, b]).max()),     # VMAS max penetration (contact on)
                    float(dep_on_off.max()), first_dep,           # contact-induced departure
                    float(dep_off_kin.max()),                     # drag/model departure (no contacts)
                    int((e_kin[b] >= 0) != (e_on[b] >= 0)),       # event outcome disagrees kin vs vmas_on
                    int((e_off[b] >= 0) != (e_on[b] >= 0)),       # event outcome disagrees off vs on
                ])
    hdr = ["case", "drag", "dt", "steps", "seed", "idx", "lateral_offset_m",
           "kin_min_sep_m", "vmas_off_min_sep_m", "vmas_on_min_sep_m",
           "kin_first_event_step", "vmas_off_first_event_step", "vmas_on_first_event_step",
           "vmas_on_max_penetration_m", "contact_departure_max_m", "contact_departure_first_step",
           "drag_model_departure_max_m", "event_mismatch_kin_vs_on", "event_mismatch_off_vs_on"]
    with open(out_dir / "partB_collision_cases.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(rows)
    return hdr, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED); np.random.seed(SEED)
    cfg = {
        "seed": SEED, "vmas": vmas.__version__ if hasattr(vmas, "__version__") else "1.5.2",
        "torch": torch.__version__, "numpy": np.__version__, "python": sys.version,
        "platform": platform.platform(),
        "assumed_bounds": {"V_MAX_mps": V_MAX, "W_MAX_radps": W_MAX,
                           "ROBOT_R_m": ROBOT_R, "OBST_R_m": OBST_R},
        "partA": {"dt": [0.1, 0.05], "drag": [0.25, 0.0], "integration": ["rk4", "euler"],
                  "cmd_modes": ["const", "rand"], "envs": 100, "robots_per_env": 3, "steps": 20,
                  "collide": False, "reference": "closed-form unicycle arc, float64"},
        "partB": {"cases": ["robot_obstacle", "robot_robot"], "drag": [0.0, 0.25], "dt": 0.1,
                  "steps": 40, "offsets": "31 lateral offsets per case",
                  "VMAS_defaults": {"collision_force": 100, "contact_margin": 1e-3, "world_drag": 0.25}},
    }
    (out / "gate_config.json").write_text(json.dumps(cfg, indent=2))
    part_a(out)
    part_b(out)
    print("done ->", out.resolve())


if __name__ == "__main__":
    main()
