"""Shared helpers for the gate and the pilot: evaluation layouts, environment factory, and one
filter-off / filter-on evaluation of a frozen policy through protocol.run_trials."""
from __future__ import annotations

import numpy as np

from .config import StudyConfig, check_paired
from .envs import VmasEnv
from .protocol import Outcome, run_trials
from .scenario import sample_layout


def eval_layout(cfg: StudyConfig, n_obst: int, kind: str, batch: int):
    seed = {"validation": cfg.validation_seed, "pilot": cfg.pilot_seed}[kind](n_obst)
    return sample_layout(seed, batch, n_obst, cfg.layout_filter_config(), cfg.scenario_config())


def vmas_factory(cfg: StudyConfig, batch: int, n_obst: int, seed: int = 0):
    fc = cfg.layout_filter_config()
    n = cfg.scenario_config().n_agents
    return lambda: VmasEnv(batch, n, n_obst, cfg.dt, fc.robot_r, fc.obst_r, fc.v_max, fc.w_max, seed=seed)


def run_condition(cfg: StudyConfig, policy, layout, n_obst, alpha=None, log_steps=False):
    """alpha=None -> filter OFF; otherwise filter ON with that barrier gain. Both conditions are
    built from the same config and checked with check_paired against the filter-off values."""
    pc = cfg.protocol_config()
    fc_off = cfg.filter_config(cfg.alphas[0])
    fc = fc_off if alpha is None else cfg.filter_config(alpha)
    check_paired(pc, fc_off, cfg.protocol_config(), fc)
    env = vmas_factory(cfg, layout.init_poses.shape[0], n_obst)()
    res = run_trials(env, policy, layout, pc, fc, use_filter=alpha is not None, log_steps=log_steps)
    if hasattr(policy, "assert_unchanged"):
        policy.assert_unchanged()
    return res


def outcome_counts(res):
    o = res.outcome
    return {Outcome(k).name: int((o == k).sum()) for k in (Outcome.GOAL, Outcome.COLLISION, Outcome.INFEASIBLE, Outcome.TIMEOUT)}


def trial_rows(res, **tags):
    """Per-trial records (one dict per trial) for CSV output."""
    rows = []
    for b in range(len(res.outcome)):
        rows.append(dict(tags, trial=b, outcome=Outcome(res.outcome[b]).name, end_step=int(res.end_step[b]),
                         collision_kind=int(res.collision_kind[b]), form_err_final=float(res.form_err_final[b]),
                         form_err_mean=float(res.form_err_mean[b]), goal_dist_final=float(res.goal_dist_final[b]),
                         n_interventions=int(res.n_interventions[b]), n_infeasible=int(res.n_infeasible[b]),
                         h_obst_min=float(res.h_obst_min[b]), h_rr_min=float(res.h_rr_min[b]),
                         stalled=int(res.stalled[b]), track_err_max=float(res.track_err_max[b])))
    return rows


def write_rows(path, rows):
    import csv
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def rate_ci(k, n):
    from .stats import wilson_ci
    lo, hi = wilson_ci(k, n)
    return k / n if n else float("nan"), lo, hi


def same_layouts(a, b):
    return np.array_equal(a.init_poses, b.init_poses) and np.array_equal(a.obst_xy, b.obst_xy)
