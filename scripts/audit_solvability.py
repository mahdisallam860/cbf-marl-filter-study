#!/usr/bin/env python3
"""Layout solvability audit on held-out layouts (validation = the gate's layouts, pilot), per
density: FOUND / BLOCKED / UNDETERMINED for (a) each robot alone, (b) the team avoiding one
another, (c) the team within the formation tolerance during transit. Geometry only; method and
proof rules in cbf_study/solvability.py. Puts the results beside the gate GOAL rates.

Writes results/solvability/{layouts.csv, summary.csv, SOLVABILITY.md, audit_config.json}."""
import argparse, csv, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def audit_chunk(args):
    split, K, batch, lo, hi, config_path = args
    from cbf_study.config import StudyConfig
    from cbf_study.evaluate import eval_layout
    from cbf_study.solvability import AuditConfig, Grid, audit_layout
    cfg = StudyConfig.load(config_path) if config_path else StudyConfig.load()
    ac = AuditConfig.from_study(cfg)
    grid = Grid(ac)
    lay = eval_layout(cfg, K, split, batch)
    seed = {"validation": cfg.validation_seed, "pilot": cfg.pilot_seed}[split](K)
    rows = []
    for b in range(lo, hi):
        o = audit_layout(lay.init_poses[b], lay.obst_xy[b], lay.goal[b], lay.slots, ac, grid)
        rows.append(dict(split=split, density=K, layout_seed=seed, trial=b, **o))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", type=int, default=200, help="layouts per density (the gate used 200)")
    ap.add_argument("--pilot", type=int, default=500, help="layouts per density")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--config", default=None)
    ap.add_argument("--gate", default="results/gate/gate_by_checkpoint.csv")
    ap.add_argument("--out", default="results/solvability")
    a = ap.parse_args()
    from cbf_study.config import StudyConfig
    from cbf_study.evaluate import write_rows
    from cbf_study.solvability import BLOCKED, FOUND, UNDETERMINED, AuditConfig
    from cbf_study.stats import wilson_ci
    cfg = StudyConfig.load(a.config) if a.config else StudyConfig.load()
    ac = AuditConfig.from_study(cfg)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tasks = []
    for split, n in (("validation", a.validation), ("pilot", a.pilot)):
        for K in cfg.densities:
            for lo in range(0, n, 25):
                tasks.append((split, K, n, lo, min(n, lo + 25), a.config))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for chunk in ex.map(audit_chunk, tasks) for r in chunk]
    write_rows(out / "layouts.csv", rows)

    summ = []
    for split in ("validation", "pilot"):
        for K in cfg.densities:
            rs = [r for r in rows if r["split"] == split and r["density"] == K]
            s = dict(split=split, density=K, layout_seed=rs[0]["layout_seed"], n=len(rs))
            for q in ("a", "b", "c"):
                st = [r[f"{q}_status"] for r in rs]
                for lab in (FOUND, BLOCKED, UNDETERMINED):
                    s[f"{q}_{lab}"] = st.count(lab)
                lo, hi = wilson_ci(s[f"{q}_{FOUND}"], len(rs))
                s[f"{q}_found_frac"], s[f"{q}_found_lo"], s[f"{q}_found_hi"] = s[f"{q}_{FOUND}"] / len(rs), lo, hi
                key = "a_steps_max" if q == "a" else f"{q}_steps"
                steps = sorted(r[key] for r in rs if r[f"{q}_status"] == FOUND)
                s[f"{q}_steps_median"] = steps[len(steps) // 2] if steps else None
                s[f"{q}_steps_max"] = steps[-1] if steps else None
            modes = [r["b_mode"].split(" ")[0] for r in rs if r["b_status"] == FOUND]
            s["b_modes"] = ";".join(f"{m}:{modes.count(m)}" for m in sorted(set(modes)))
            summ.append(s)
    write_rows(out / "summary.csv", summ)
    meta = dict(audit_config=asdict(ac), env_fingerprint=cfg.env_fingerprint(), horizon_s=ac.max_steps * ac.dt,
                wall_s=round(time.time() - t0, 1))
    (out / "audit_config.json").write_text(json.dumps(meta, indent=1))

    # gate goal rates beside the audit (validation layouts are the gate's layouts)
    gate = list(csv.DictReader(open(a.gate))) if Path(a.gate).exists() else []
    sel = json.loads(Path(a.gate).with_name("selection.json").read_text()) if gate else None
    L = ["# Layout solvability audit (geometry only) beside the filter-off gate", "",
         f"Planner: grid search (cell {ac.h} m) on certainly-free cells, shortcut, every segment checked exactly; "
         f"obstacles inflated by R + R_o + {ac.margin} m, robots kept {2 * cfg.protocol_config().robot_r} + {ac.margin} m apart; "
         f"unicycle follows the polyline by turning in place (|w| <= {ac.w_max} rad/s) and driving straight "
         f"(v <= {ac.v_max} m/s) at dt = {ac.dt} s; horizon {ac.max_steps} steps ({ac.max_steps * ac.dt:.0f} s). "
         f"Formation tolerance {ac.formation_tol} m mean slot error (the protocol checks it only at the goal; (c) "
         f"imposes it during the whole transit). BLOCKED = proven by disconnection of an over-approximated free "
         f"space (any horizon); UNDETERMINED = neither a verified plan nor a proof. No arena walls (A6). "
         f"Audit box {ac.box}, outer ring free.", "",
         "| split | K | layouts (seed) | (a) each robot found / blocked / undet. | (b) team found / blocked / undet. | "
         "(c) team within tol found / blocked / undet. | (b) steps median / max | gate GOAL, selected ckpt | gate GOAL, best ckpt |",
         "|---|---|---|---|---|---|---|---|---|"]
    for s in summ:
        g_sel = g_best = "-"
        if gate and s["split"] == "validation":
            dk = [r for r in gate if r["density"] == str(s["density"])]
            best = max(dk, key=lambda r: float(r["goal_rate"]))
            selr = next(r for r in dk if r["ckpt"] == sel["best"]["ckpt"])
            g_sel = f"{100 * float(selr['goal_rate']):.1f}%"
            g_best = (f"{100 * float(best['goal_rate']):.1f}% (seed {best['seed']} it {best['iteration']})"
                      if float(best["goal_rate"]) > 0 else f"0.0% (all {len(dk)} checkpoints)")
        trip = lambda q: (f"{s[q + '_found']} ({100 * s[q + '_found_frac']:.1f}%, CI {100 * s[q + '_found_lo']:.1f}-"
                          f"{100 * s[q + '_found_hi']:.1f}) / {s[q + '_blocked']} / {s[q + '_undetermined']}")
        L.append(f"| {s['split']} | {s['density']} | {s['n']} ({s['layout_seed']}) | {trip('a')} | {trip('b')} | {trip('c')} | "
                 f"{s['b_steps_median']} / {s['b_steps_max']} | {g_sel} | {g_best} |")
    (out / "SOLVABILITY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
