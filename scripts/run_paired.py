#!/usr/bin/env python3
"""Paired filter-off / filter-on run on identical seeded layouts, with the STAND-IN policy.
Writes summary and per-step CSVs. Replace SlotPolicy with the frozen MAPPO actor for the study."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbf_study.cbf_qp import FilterConfig
from cbf_study.envs import KinematicEnv, VmasEnv
from cbf_study.protocol import ProtocolConfig, run_paired, write_step_csv, write_summary_csv, Outcome
from cbf_study.scenario import SlotPolicy, sample_layout

ap = argparse.ArgumentParser()
ap.add_argument("--sim", choices=["vmas", "kinematic"], default="vmas")
ap.add_argument("--n-obst", type=int, default=5)
ap.add_argument("--batch", type=int, default=100)
ap.add_argument("--alpha", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=20261025)
ap.add_argument("--steps", type=int, default=400)
ap.add_argument("--out", default="results/paired")
ap.add_argument("--log-trials", type=int, default=3, help="per-step CSV for the first N trials")
a = ap.parse_args()

dt = 0.1
fc = FilterConfig(dt=dt, robot_r=0.10, obst_r=0.20, ell=0.05, v_min=-0.5, v_max=0.5, w_max=1.0,
                  alpha=a.alpha, sensing_range=1.0, comm_range=1.5)
pc = ProtocolConfig(dt=dt, robot_r=0.10, obst_r=0.20, v_min=-0.5, v_max=0.5, w_max=1.0,
                    max_steps=a.steps, goal_tol=0.10, formation_tol=0.10)
lay = sample_layout(a.seed, a.batch, a.n_obst, fc)
fac = (lambda: VmasEnv(a.batch, 3, a.n_obst, dt, fc.robot_r, fc.obst_r, fc.v_max, fc.w_max, seed=a.seed)) \
    if a.sim == "vmas" else (lambda: KinematicEnv(dt))
off, on = run_paired(fac, SlotPolicy(pc.v_max, pc.w_max), lay, pc, fc)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for name, r in (("off", off), ("on", on)):
    write_summary_csv(out / f"summary_{name}.csv", r, label=f"K{a.n_obst}_a{a.alpha}")
    write_step_csv(out / f"steps_{name}.csv", r, trials=range(min(a.log_trials, a.batch)), label=f"K{a.n_obst}_a{a.alpha}")
    print(name, {Outcome(o).name: int((r.outcome == o).sum()) for o in (1, 2, 3, 4)},
          "interventions", int(r.n_interventions.sum()))
