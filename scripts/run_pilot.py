#!/usr/bin/env python3
"""Pilot: paired filter-off / filter-on runs of the frozen MAPPO policy on the PILOT layouts at
every density and every alpha in the sweep; outcomes, formation error, interventions, stalled
(deadlock) flag; paired tests with CIs and the sample size needed for the McNemar and Wilcoxon
tests. Refuses to run unless the gate in results/gate/selection.json passed.

Filter-off is run once per density (it does not depend on alpha; checked) and paired with every
filter-on run on the identical layouts. Infeasible QP steps end the trial as INFEASIBLE and are
reported, never repaired.

Writes results/pilot/<tag>/{trials.csv, cells.csv, sample_size.csv, steps_*.csv, meta.json}."""
import argparse, hashlib, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def layout_hash(lay):
    h = hashlib.sha256()
    for a in (lay.init_poses, lay.obst_xy, lay.goal, lay.slots):
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def run_task(args):
    ckpt, K, alpha, batch, step_trials, out = args
    import torch
    torch.set_num_threads(1)
    from cbf_study.config import StudyConfig
    from cbf_study.evaluate import eval_layout, run_condition, trial_rows
    from cbf_study.policies import FrozenMAPPOPolicy
    from cbf_study.protocol import write_step_csv
    cfg = StudyConfig.load()
    pol = FrozenMAPPOPolicy(ckpt, cfg)
    lay = eval_layout(cfg, K, "pilot", batch)
    t0 = time.time()
    res = run_condition(cfg, pol, lay, K, alpha=alpha, log_steps=step_trials > 0)
    tag = "off" if alpha is None else f"a{alpha:g}"
    if step_trials:
        write_step_csv(Path(out) / f"steps_K{K}_{tag}.csv", res, trials=range(step_trials), label=f"K{K}_{tag}")
    rows = trial_rows(res, density=K, filter=int(alpha is not None), alpha="off" if alpha is None else alpha,
                      layout=layout_hash(lay))
    return dict(K=K, alpha=alpha, rows=rows, wall=time.time() - t0, digest=pol.digest)


def cell_stats(off, on, n_agents, alpha_bonf):
    import numpy as np
    from cbf_study import stats as S
    o_off = np.array([r["outcome"] for r in off]); o_on = np.array([r["outcome"] for r in on])
    st_off = np.array([r["stalled"] for r in off], bool); st_on = np.array([r["stalled"] for r in on], bool)
    fe_off = np.array([r["form_err_mean"] for r in off]); fe_on = np.array([r["form_err_mean"] for r in on])
    both_goal = (o_off == "GOAL") & (o_on == "GOAL")
    end_on = np.array([r["end_step"] for r in on])
    interv = np.array([r["n_interventions"] for r in on])
    cell = dict(n=len(off))
    for name in ("GOAL", "COLLISION", "INFEASIBLE", "TIMEOUT"):
        cell[f"off_{name}"] = int((o_off == name).sum()); cell[f"on_{name}"] = int((o_on == name).sum())
    cell["off_stalled"], cell["on_stalled"] = int(st_off.sum()), int(st_on.sum())
    cell["interv_per_trial"] = float(interv.mean())
    cell["interv_frac"] = float(interv.sum() / max(1, (end_on * n_agents).sum()))
    cell["on_h_obst_min"] = float(min(r["h_obst_min"] for r in on)); cell["on_h_rr_min"] = float(min(r["h_rr_min"] for r in on))
    ss = []
    for ep, ev_off, ev_on in (("collision", o_off == "COLLISION", o_on == "COLLISION"),
                              ("goal", o_off == "GOAL", o_on == "GOAL"),
                              ("stalled", st_off, st_on)):
        m = S.mcnemar_summary(ev_off, ev_on)
        for k in ("n10", "n01", "diff", "diff_lo", "diff_hi", "p_value", "var_diff"):
            cell[f"{ep}_{k}"] = m[k]
        p10, p01 = m["n10"] / m["n"], m["n01"] / m["n"]
        for lab, a in (("a05", 0.05), ("bonf", alpha_bonf)):
            n_c = S.mcnemar_n_connor(p10, p01, alpha=a)
            n_s = S.min_n_for_power(lambda n: S.mcnemar_power_sim(p10, p01, n, alpha=a, reps=4000, seed=11)) if np.isfinite(n_c) else float("inf")
            ss.append(dict(endpoint=ep, test="McNemar", alpha_level=a, label=lab, effect=m["diff"],
                           p10=p10, p01=p01, n_closed_form=n_c, n_simulated=n_s))
    for ep, sel in (("form_err_all", np.ones(len(off), bool)), ("form_err_both_goal", both_goal)):
        if sel.sum() < 10:
            cell[f"{ep}_n"] = int(sel.sum())
            continue
        w = S.wilcoxon_summary(fe_off[sel], fe_on[sel])
        for k in ("n", "n_nonzero", "mean_off", "mean_on", "mean_diff", "mean_diff_lo", "mean_diff_hi", "sd_diff",
                  "hl", "hl_lo", "hl_hi", "p_value", "p_prime"):
            cell[f"{ep}_{k}"] = w[k]
        d = fe_on[sel] - fe_off[sel]
        for lab, a in (("a05", 0.05), ("bonf", alpha_bonf)):
            n_c = S.wilcoxon_n_noether(d, alpha=a)
            n_s = S.min_n_for_power(lambda n: S.wilcoxon_power_sim(d, n, alpha=a, reps=2000, seed=12)) if np.isfinite(n_c) else float("inf")
            ss.append(dict(endpoint=ep, test="Wilcoxon", alpha_level=a, label=lab, effect=w["mean_diff"],
                           p10=float("nan"), p01=float("nan"), n_closed_form=n_c, n_simulated=n_s))
    return cell, ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=500, help="pilot pairs per density")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--step-trials", type=int, default=2, help="per-step CSV for the first N trials")
    ap.add_argument("--gate", default="results/gate/selection.json")
    ap.add_argument("--out", default="results/pilot")
    a = ap.parse_args()
    from cbf_study.config import StudyConfig
    from cbf_study.evaluate import write_rows
    cfg = StudyConfig.load()
    gate = json.loads(Path(a.gate).read_text())
    passed = {gate["best"]["ckpt"]: gate["best"]["gate_pass"],
              **{v["ckpt"]: v["gate_pass"] for v in gate["best_per_seed"].values()}}
    if not passed.get(a.ckpt, False):
        sys.exit(f"STOP: {a.ckpt} did not pass the gate (or was not selected); no filter result is produced.")

    out = Path(a.out) / a.tag
    out.mkdir(parents=True, exist_ok=True)
    tasks = [(a.ckpt, K, al, a.batch, a.step_trials, str(out)) for K in cfg.densities for al in (None,) + cfg.alphas]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(run_task, tasks))
    rows = [r for res in results for r in res["rows"]]
    write_rows(out / "trials.csv", rows)
    assert len({res["digest"] for res in results}) == 1, "policy parameters differ between runs"

    m_tests = len(cfg.densities) * len(cfg.alphas)
    cells, sizes = [], []
    for K in cfg.densities:
        off = next(r for r in results if r["K"] == K and r["alpha"] is None)["rows"]
        for al in cfg.alphas:
            on = next(r for r in results if r["K"] == K and r["alpha"] == al)["rows"]
            assert {r["layout"] for r in off} == {r["layout"] for r in on}, "pairs are not on identical layouts"
            cell, ss = cell_stats(off, on, cfg.scenario_config().n_agents, 0.05 / m_tests)
            cells.append(dict(density=K, alpha=al, alpha_dt=al * cfg.dt, **cell))
            sizes += [dict(density=K, alpha=al, **s) for s in ss]
            print(f"K{K} a{al:g}: off G/C/I/T {cell['off_GOAL']}/{cell['off_COLLISION']}/{cell['off_INFEASIBLE']}/{cell['off_TIMEOUT']} "
                  f"on {cell['on_GOAL']}/{cell['on_COLLISION']}/{cell['on_INFEASIBLE']}/{cell['on_TIMEOUT']} "
                  f"stalled on {cell['on_stalled']} interv/trial {cell['interv_per_trial']:.1f}", flush=True)
    write_rows(out / "cells.csv", cells)
    write_rows(out / "sample_size.csv", sizes)
    meta = dict(ckpt=a.ckpt, batch=a.batch, densities=cfg.densities, alphas=cfg.alphas, policy_digest=results[0]["digest"],
                env_fingerprint=cfg.env_fingerprint(), pending_placeholders=len(cfg.pending()),
                bonferroni_family=m_tests, wall_s=round(time.time() - t0, 1),
                task_wall_s={f"K{r['K']}_{'off' if r['alpha'] is None else r['alpha']}": round(r["wall"], 1) for r in results})
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
