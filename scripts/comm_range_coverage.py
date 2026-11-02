#!/usr/bin/env python3
"""Descriptive check of how often the filter's neighbour range (comm_range) and obstacle range
(sensing_range) actually excluded anything in the final test (filter-on arm, saved poses only).

For every executed filter-on period (pre-step state) of every final-test trial: the barrier-point
distance of each robot pair and the largest pair distance; the share of robot-pair-periods with
distance above comm_range. Writes results/paper_revision/comm_range_coverage.json. No simulation."""
import json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbf_study.config import StudyConfig      # noqa: E402

FT = ROOT / "results" / "final_test"


def main():
    fc = StudyConfig.load().filter_config(10.0)
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())
    n_pp, n_out, dmax = 0, 0, 0.0
    for p in sel["policies"]:
        for K in (1, 2, 3):
            z = np.load(FT / "steps" / f"{p['policy_id']}_K{K}_on.npz")
            po, T, P = z["pose_offsets"], z["executed_periods"], z["poses"]
            for t in range(len(T)):
                s = P[po[t]:po[t] + int(T[t])].astype(np.float64)       # pre-step states of executed periods
                if not len(s):
                    continue
                pb = s[..., :2] + fc.ell * np.stack([np.cos(s[..., 2]), np.sin(s[..., 2])], -1)
                for i, j in ((0, 1), (0, 2), (1, 2)):
                    d = np.linalg.norm(pb[:, i] - pb[:, j], axis=1)
                    n_pp += len(d)
                    n_out += int((d > fc.comm_range).sum())
                    dmax = max(dmax, float(d.max()))
    out = dict(label="descriptive", comm_range=fc.comm_range, sensing_range=fc.sensing_range,
               pair_periods=n_pp, pair_periods_beyond_comm_range=n_out, max_pair_distance=dmax,
               closing_per_period_pair=2 * fc.vp * fc.dt, closing_per_period_obstacle=fc.vp * fc.dt,
               d_rr=fc.d_rr, d_obst=fc.d_obst)
    (ROOT / "results/paper_revision/comm_range_coverage.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
