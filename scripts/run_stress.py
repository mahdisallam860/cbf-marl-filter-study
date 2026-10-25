#!/usr/bin/env python3
"""Adversarial stress runs (filter on vs off). Writes results/stress_summary.csv.
Evidence consistent with the one-period lemma; NOT a proof of continuous-time safety.
Radii, limits, ell, ranges and tolerances come from config/study.toml; the (dt, alpha) grid and the
horizon below are stress-test overrides, not study values."""
import argparse, csv, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbf_study.config import StudyConfig, check_paired
from cbf_study.envs import KinematicEnv, VmasEnv
from cbf_study.stress import run_stress, summarise

def cfgs(dt, alpha, T):
    fc = replace(CFG.filter_config(alpha), dt=dt)
    fc.validate()
    pc = replace(CFG.protocol_config(), dt=dt, max_steps=T)
    check_paired(pc, fc, pc, fc)
    return fc, pc

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="results/stress_summary.csv")
ap.add_argument("--batch", type=int, default=100)
ap.add_argument("--config", default=None)
a = ap.parse_args()
CFG = StudyConfig.load(a.config) if a.config else StudyConfig.load()
rows = []
for sim in ("kinematic", "vmas"):
    for dt, alpha in ((0.1, 0.5), (0.1, 2.0), (0.1, 5.0), (0.05, 2.0), (0.05, 10.0)):
        if sim == "vmas" and (dt, alpha) not in ((0.1, 2.0), (0.05, 2.0)):
            continue
        fc, pc = cfgs(dt, alpha, int(25 / dt))
        for K in (2, 5, 8):
            for which in ("attack", "random"):
                fac = (lambda: KinematicEnv(dt)) if sim == "kinematic" else \
                      (lambda: VmasEnv(a.batch, CFG.scenario_config().n_agents, K, dt, fc.robot_r, fc.obst_r, fc.v_max, fc.w_max, seed=1))
                off, on = run_stress(fac, pc, fc, K, a.batch, 1000 + K, which)
                r = dict(sim=sim, dt=dt, alpha=alpha, n_obst=K, policy=which, **summarise(off, on))
                rows.append(r); print(r, flush=True)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
