#!/usr/bin/env python3
"""Rebuild curve.csv rows and completed-stage summaries of a curriculum run from its train.log,
for runs started before train_curriculum.py wrote its tables incrementally (seed 1 crashed in the
final stage on a sampler stall, losing the in-memory tables). Only iterations up to the state
saved in latest.pt are kept; the summaries are inserted into latest.pt so that --resume writes a
complete stages.json. Held-out rows are regenerated separately by re-evaluating the saved stage
checkpoints (--heldout), which also checks them against the values in the log.

Reconstructed curve rows carry source=train.log; fields the log does not contain are empty."""
import argparse, csv, json, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch

IT = re.compile(r"^stage (\d+) it (\d+) steps (\d+)/(\d+) total (\d+) goal ([\d.na]+) coll ([\d.na]+) timeout ([\d.na]+) kl ([\d.na]+)")
ST = re.compile(r"^stage (\d+) densities \(([\d, ]+)\) budget (\d+)")
HO = re.compile(r"K(\d+): goal ([\d.]+) coll ([\d.]+) timeout ([\d.]+)")
COLS = ["stage", "densities", "stage_env_steps", "iteration", "env_steps", "wall_s", "episodes", "goal_rate",
        "collision_rate", "timeout_rate", "ep_return", "ep_len", "form_err_mean", "approx_kl", "clipfrac", "ratio0_maxdev"]


def parse(log_path, max_iteration):
    curve, stage_info, heldout_log = [], {}, []
    g = 0
    cur_stage = None
    for line in open(log_path):
        line = line.rstrip("\n")
        if line.startswith("resumed"):
            break
        m = ST.match(line)
        if m:
            cur_stage = int(m.group(1))
            stage_info[cur_stage] = dict(densities=[int(x) for x in m.group(2).replace(" ", "").split(",") if x],
                                         budget=int(m.group(3)))
            continue
        m = IT.match(line)
        if m:
            g += 1
            if g > max_iteration:
                break
            si, it, steps, budget, total = (int(m.group(i)) for i in range(1, 6))
            curve.append(dict(stage=si, densities="/".join(map(str, stage_info[si]["densities"])), stage_env_steps=steps,
                              iteration=g, env_steps=total, goal_rate=m.group(6), collision_rate=m.group(7),
                              timeout_rate=m.group(8), approx_kl=m.group(9), source="train.log"))
            stage_info[si].update(env_steps_used=steps, cumulative_env_steps=total, iterations=it, last_iteration=g)
            continue
        if line.strip().startswith("held-out") and curve:
            ev = {int(k): dict(goal=float(a), collision=float(b), timeout=float(c)) for k, a, b, c in HO.findall(line)}
            heldout_log.append(dict(iteration=curve[-1]["iteration"], stage=curve[-1]["stage"], ev=ev))
            stage_info[curve[-1]["stage"]]["heldout_end"] = ev
    return curve, stage_info, heldout_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="results/curriculum/seed_<s>")
    ap.add_argument("--heldout", action="store_true", help="re-evaluate saved stage checkpoints into heldout_rebuilt.csv")
    a = ap.parse_args()
    run = Path(a.run)
    st = torch.load(run / "latest.pt", map_location="cpu", weights_only=False)
    max_it, cur_stage = st["iteration"], st["stage"]
    curve, info, ho_log = parse(run / "train.log", max_it)
    if not a.heldout:
        assert not (run / "curve.csv").exists(), "curve.csv exists; nothing to rebuild"
        with open(run / "curve.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS + ["source"]); w.writeheader()
            w.writerows({k: r.get(k, "") for k in COLS + ["source"]} for r in curve)
        stages = [dict(stage=si, densities=info[si]["densities"], budget=info[si]["budget"],
                       env_steps_used=info[si]["env_steps_used"], cumulative_env_steps=info[si]["cumulative_env_steps"],
                       iterations=info[si]["iterations"], advanced_early=info[si]["env_steps_used"] < info[si]["budget"],
                       heldout_end=info[si]["heldout_end"], wall_s=None, source="train.log")
                  for si in sorted(info) if si < cur_stage]
        st["stages"] = stages
        torch.save(st, run / "latest.pt")
        print(f"rebuilt {len(curve)} curve rows (iterations <= {max_it}) and {len(stages)} stage summaries")
        print(json.dumps([(s["stage"], s["densities"], s["env_steps_used"], s["cumulative_env_steps"]) for s in stages]))
        return
    from cbf_study.config import StudyConfig
    from train_curriculum import heldout
    cfg = StudyConfig.load()
    cur = cfg.curriculum()
    rows, mism = [], 0
    for h in ho_log:
        if h["iteration"] > max_it:
            continue
        cks = sorted((run / "stage_checkpoints").glob(f"stage{h['stage']}_K*_it{h['iteration']:04d}.pt"))
        assert len(cks) == 1, (h, cks)
        ev = heldout(cfg, cks[0], list(h["ev"]), cur["eval_trials"])
        c = next(r for r in curve if r["iteration"] == h["iteration"])
        for K, e in ev.items():
            mism += any(abs(e[k] - h["ev"][K][k]) > 1e-9 for k in ("goal", "collision", "timeout"))
            rows.append(dict(stage=h["stage"], densities=c["densities"], iteration=h["iteration"],
                             stage_env_steps=c["stage_env_steps"], env_steps=c["env_steps"], density=K, ckpt=str(cks[0]), **e))
    with open(run / "heldout_rebuilt.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"re-evaluated {len(rows)} held-out rows; mismatches with the logged values: {mism}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
