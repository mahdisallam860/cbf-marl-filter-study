#!/usr/bin/env python3
"""Adversarial stress runs (filter on vs off). Writes results/stress_summary.csv.
Evidence consistent with the one-period lemma; NOT a proof of continuous-time safety."""
import argparse, csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbf_study.envs import KinematicEnv, VmasEnv
from cbf_study.stress import run_stress, summarise
from cbf_study.cbf_qp import FilterConfig
from cbf_study.protocol import ProtocolConfig

def cfgs(dt, alpha, T):
    fc = FilterConfig(dt=dt, robot_r=0.1, obst_r=0.2, ell=0.05, v_min=-0.5, v_max=0.5, w_max=1.0,
                      alpha=alpha, sensing_range=1.0, comm_range=1.5)
    pc = ProtocolConfig(dt=dt, robot_r=0.1, obst_r=0.2, v_min=-0.5, v_max=0.5, w_max=1.0,
                        max_steps=T, goal_tol=0.1, formation_tol=0.1)
    return fc, pc

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/stress_summary.csv")
ap.add_argument("--batch", type=int, default=100)
a = ap.parse_args()
rows = []
for sim in ("kinematic", "vmas"):
    for dt, alpha in ((0.1, 0.5), (0.1, 2.0), (0.1, 5.0), (0.05, 2.0), (0.05, 10.0)):
        if sim == "vmas" and (dt, alpha) not in ((0.1, 2.0), (0.05, 2.0)):
            continue
        fc, pc = cfgs(dt, alpha, int(25 / dt))
        for K in (2, 5, 8):
            for which in ("attack", "random"):
                fac = (lambda: KinematicEnv(dt)) if sim == "kinematic" else \
                      (lambda: VmasEnv(a.batch, 3, K, dt, fc.robot_r, fc.obst_r, fc.v_max, fc.w_max, seed=1))
                off, on = run_stress(fac, pc, fc, K, a.batch, 1000 + K, which)
                r = dict(sim=sim, dt=dt, alpha=alpha, n_obst=K, policy=which, **summarise(off, on))
                rows.append(r); print(r, flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
