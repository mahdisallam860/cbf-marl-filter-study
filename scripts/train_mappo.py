#!/usr/bin/env python3
"""Train one MAPPO seed (filter OFF) on the VMAS formation-navigation task.
Writes results/train/seed_<s>/{curve.csv, run.json, train.log, checkpoints/ckpt_itNNNN.pt}.
Hyperparameters come from [mappo] in config/study.toml; command-line values override them."""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cbf_study.config import StudyConfig
from cbf_study.mappo import train

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--out", default=None)
ap.add_argument("--config", default=None)
ap.add_argument("--iterations", type=int, default=None)
ap.add_argument("--time-limit-min", type=float, default=None)
ap.add_argument("--envs-per-density", type=int, default=None)
ap.add_argument("--resume", action="store_true")
ap.add_argument("--densities", type=int, nargs="+", default=None,
                help="DIAGNOSTIC ONLY: train on these obstacle counts instead of the study densities")
ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                help="override any [mappo] value, e.g. --set epochs=10 (recorded per training phase)")
a = ap.parse_args()
cfg = StudyConfig.load(a.config) if a.config else StudyConfig.load()
ov = {k: v for k, v in (("max_iterations", a.iterations), ("time_limit_min", a.time_limit_min),
                        ("envs_per_density", a.envs_per_density)) if v is not None}
for kv in a.set:
    k, v = kv.split("=", 1)
    if k not in cfg.mappo:
        sys.exit(f"unknown [mappo] key {k}")
    ov[k] = type(cfg.mappo[k])(v)
out = Path(a.out or f"results/train/seed_{a.seed}")
out.mkdir(parents=True, exist_ok=True)
logf = open(out / "train.log", "a")
def log(msg):
    print(msg, flush=True); logf.write(msg + "\n"); logf.flush()
log(f"seed {a.seed} env_fingerprint {cfg.env_fingerprint()} pending_placeholders {len(cfg.pending())} overrides {ov}")
if a.densities is not None and "results/train/" in str(out).replace("\\", "/"):
    sys.exit("diagnostic runs (--densities) must not be written under results/train/ (the gate reads every checkpoint there)")
train(cfg, a.seed, out, overrides=ov, resume=a.resume, log=log, densities=a.densities)
