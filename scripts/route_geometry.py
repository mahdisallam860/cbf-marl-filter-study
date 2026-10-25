#!/usr/bin/env python3
"""Shape of the verified routes on the gate's validation layouts: does the (near-)shortest verified
route go THROUGH the obstacle field or AROUND it? Diagnostic for the solvability audit.

For each layout: (c) rigid-formation centroid route (formation tolerance kept during transit) and
(a) each robot's individual route: length relative to the straight start-goal distance and the
largest |y| reached. Obstacle centres lie in |y| <= 0.9 m (surfaces <= 1.1 m); a centroid with
|y| >= 1.1 + slot radius + R = 1.5 m has the whole triangle outside the obstacle band.
Writes results/solvability/route_geometry.csv and route_geometry.md."""
import argparse, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np


def work(args):
    K, n, lo, hi = args
    from cbf_study.config import StudyConfig
    from cbf_study.evaluate import eval_layout
    from cbf_study.solvability import AuditConfig, Grid, audit_layout, path_length
    cfg = StudyConfig.load()
    ac = AuditConfig.from_study(cfg)
    grid = Grid(ac)
    lay = eval_layout(cfg, K, "validation", n)
    sc = cfg.scenario_config()
    band = max(abs(sc.obst_y[0]), abs(sc.obst_y[1])) + cfg.protocol_config().obst_r
    around_y = band + sc.slot_radius + cfg.protocol_config().robot_r
    rows = []
    for b in range(lo, hi):
        o = audit_layout(lay.init_poses[b], lay.obst_xy[b], lay.goal[b], lay.slots, ac, grid, return_paths=True)
        c0 = lay.init_poses[b, :, :2].mean(0)
        straight = float(np.linalg.norm(lay.goal[b] - c0))
        P = o["c_path"]
        r = dict(density=K, trial=b, c_status=o["c_status"])
        if P is not None:
            r.update(c_len_ratio=path_length(P) / straight, c_max_abs_y=float(np.abs(P[:, 1]).max()),
                     c_around=int(np.abs(P[:, 1]).max() >= around_y))
        ap = [p for p in o["a_paths"] if p is not None]
        if len(ap) == len(o["a_paths"]):
            starts = lay.init_poses[b, :, :2]
            goals = lay.goal[b] + lay.slots
            ratios = [path_length(p) / np.linalg.norm(g - s) for p, s, g in zip(ap, starts, goals)]
            r.update(a_len_ratio_max=float(max(ratios)), a_max_abs_y=float(max(np.abs(p[:, 1]).max() for p in ap)),
                     a_around=int(any(np.abs(p[:, 1]).max() >= band + cfg.protocol_config().robot_r for p in ap)))
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", default="results/solvability")
    a = ap.parse_args()
    from cbf_study.config import StudyConfig
    from cbf_study.evaluate import write_rows
    cfg = StudyConfig.load()
    tasks = [(K, a.n, lo, min(a.n, lo + 25)) for K in cfg.densities for lo in range(0, a.n, 25)]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for chunk in ex.map(work, tasks) for r in chunk]
    write_rows(Path(a.out) / "route_geometry.csv", rows)
    L = ["# Shape of the verified routes (validation layouts, the gate's layouts)", "",
         "Near-shortest verified routes from the solvability planner. 'around' = the rigid triangle's centroid "
         "reaches |y| >= 1.5 m (whole formation outside the obstacle band) / an individual robot leaves the band "
         "(|y| >= 1.2 m). Length ratio = route length / straight start-goal distance.", "",
         "| K | layouts | (c) formation routes around the field | (c) length ratio median [p90] | (a) some robot goes around | (a) worst robot length ratio median [p90] |",
         "|---|---|---|---|---|---|"]
    for K in cfg.densities:
        rs = [r for r in rows if r["density"] == K]
        c = [r for r in rs if "c_len_ratio" in r]
        aa = [r for r in rs if "a_len_ratio_max" in r]
        q = lambda xs, p: float(np.quantile(xs, p)) if xs else float("nan")
        L.append(f"| {K} | {len(rs)} | {sum(r['c_around'] for r in c)}/{len(c)} | "
                 f"{q([r['c_len_ratio'] for r in c], 0.5):.2f} [{q([r['c_len_ratio'] for r in c], 0.9):.2f}] | "
                 f"{sum(r['a_around'] for r in aa)}/{len(aa)} | "
                 f"{q([r['a_len_ratio_max'] for r in aa], 0.5):.2f} [{q([r['a_len_ratio_max'] for r in aa], 0.9):.2f}] |")
    (Path(a.out) / "route_geometry.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
