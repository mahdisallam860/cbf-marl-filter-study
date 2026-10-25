#!/usr/bin/env python3
"""Paired filter-off / filter-on run on identical seeded layouts. Writes summary and per-step CSVs.
All values come from config/study.toml. --ckpt runs the frozen MAPPO policy; without it the
obstacle-blind STAND-IN policy is used, and no result from it belongs in the study."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbf_study.config import StudyConfig, check_paired
from cbf_study.envs import KinematicEnv, VmasEnv
from cbf_study.protocol import run_paired, write_step_csv, write_summary_csv, Outcome
from cbf_study.scenario import SlotPolicy, sample_layout

ap = argparse.ArgumentParser()
ap.add_argument("--sim", choices=["vmas", "kinematic"], default="vmas")
ap.add_argument("--n-obst", type=int, default=5)
ap.add_argument("--batch", type=int, default=100)
ap.add_argument("--alpha", type=float, default=2.0)
ap.add_argument("--seed", type=int, default=20261025)
ap.add_argument("--ckpt", default=None, help="frozen MAPPO checkpoint; omit for the stand-in policy")
ap.add_argument("--config", default=None)
ap.add_argument("--out", default="results/paired")
ap.add_argument("--log-trials", type=int, default=3, help="per-step CSV for the first N trials")
a = ap.parse_args()

cfg = StudyConfig.load(a.config) if a.config else StudyConfig.load()
fc, pc = cfg.filter_config(a.alpha), cfg.protocol_config()
check_paired(pc, fc, pc, fc)
lay = sample_layout(a.seed, a.batch, a.n_obst, cfg.layout_filter_config(), cfg.scenario_config())
n = cfg.scenario_config().n_agents
fac = (lambda: VmasEnv(a.batch, n, a.n_obst, pc.dt, fc.robot_r, fc.obst_r, fc.v_max, fc.w_max, seed=a.seed)) \
    if a.sim == "vmas" else (lambda: KinematicEnv(pc.dt))
if a.ckpt:
    from cbf_study.policies import FrozenMAPPOPolicy
    pol = FrozenMAPPOPolicy(a.ckpt, cfg)
else:
    print("WARNING: stand-in SlotPolicy (obstacle-blind test double); not a study result")
    pol = SlotPolicy(pc.v_max, pc.w_max)
off, on = run_paired(fac, pol, lay, pc, fc)
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
for name, r in (("off", off), ("on", on)):
    write_summary_csv(out / f"summary_{name}.csv", r, label=f"K{a.n_obst}_a{a.alpha}")
    write_step_csv(out / f"steps_{name}.csv", r, trials=range(min(a.log_trials, a.batch)), label=f"K{a.n_obst}_a{a.alpha}")
    print(name, {Outcome(o).name: int((r.outcome == o).sum()) for o in (1, 2, 3, 4)},
          "interventions", int(r.n_interventions.sum()))
