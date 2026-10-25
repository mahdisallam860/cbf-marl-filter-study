#!/usr/bin/env python3
"""Gate before any filter result: evaluate every checkpoint with the filter OFF at each density on
the VALIDATION layouts (disjoint from training and pilot layouts), report GOAL and COLLISION rates
with Wilson 95% CIs, apply the pre-registered [gate] thresholds from config/study.toml, and select
the best checkpoint by the pre-registered rule (highest pooled GOAL rate; ties -> later checkpoint).

Writes results/gate/{gate_by_checkpoint.csv, selection.json, GATE.md}."""
import argparse, glob, json, os, re, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def eval_ckpt(args):
    path, batch = args
    import torch
    torch.set_num_threads(1)
    from cbf_study.config import StudyConfig
    from cbf_study.evaluate import eval_layout, outcome_counts, run_condition
    from cbf_study.policies import FrozenMAPPOPolicy
    cfg = StudyConfig.load()
    pol = FrozenMAPPOPolicy(path, cfg)
    m = re.search(r"seed_?(\d+).*ckpt_it(\d+)", path)
    rows = []
    for K in cfg.densities:
        lay = eval_layout(cfg, K, "validation", batch)
        res = run_condition(cfg, pol, lay, K, alpha=None)
        c = outcome_counts(res)
        rows.append(dict(seed=int(m.group(1)), iteration=int(m.group(2)), env_steps=pol.meta["env_steps"],
                         density=K, eval_split="validation", eval_layout_seed=str(cfg.validation_seed(K)), n=batch, **c, stalled=int(res.stalled.sum()),
                         form_err_mean=float(res.form_err_mean.mean()),
                         goal_steps_mean=float(res.end_step[res.outcome == 1].mean()) if c["GOAL"] else float("nan"),
                         ckpt=path))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="results/train/seed_*/checkpoints/ckpt_it*.pt")
    ap.add_argument("--batch", type=int, default=200, help="validation trials per density")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="results/gate")
    a = ap.parse_args()
    from cbf_study.config import StudyConfig
    from cbf_study.evaluate import rate_ci, write_rows
    cfg = StudyConfig.load()
    paths = sorted(glob.glob(a.ckpts))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(eval_ckpt, [(p, a.batch) for p in paths]):
            rows += r
            print(f"{r[0]['ckpt']}: " + "  ".join(f"K{x['density']} goal {x['GOAL']}/{x['n']} coll {x['COLLISION']}/{x['n']}" for x in r),
                  flush=True)
    gmin, cmin = float(cfg["gate.goal_rate_min"]), float(cfg["gate.collision_rate_min"])
    table = []
    for (seed, it) in sorted({(r["seed"], r["iteration"]) for r in rows}):
        rs = [r for r in rows if r["seed"] == seed and r["iteration"] == it]
        pooled = dict(rs[0], density="pooled", n=sum(r["n"] for r in rs),
                      eval_layout_seed="+".join(r["eval_layout_seed"] for r in rs),
                      **{k: sum(r[k] for r in rs) for k in ("GOAL", "COLLISION", "INFEASIBLE", "TIMEOUT", "stalled")})
        pooled["form_err_mean"] = sum(r["form_err_mean"] * r["n"] for r in rs) / pooled["n"]
        pooled["goal_steps_mean"] = float("nan")
        for r in rs + [pooled]:
            g, glo, ghi = rate_ci(r["GOAL"], r["n"])
            c, clo, chi = rate_ci(r["COLLISION"], r["n"])
            r.update(goal_rate=g, goal_lo=glo, goal_hi=ghi, coll_rate=c, coll_lo=clo, coll_hi=chi,
                     timeout_rate=r["TIMEOUT"] / r["n"], goal_ok=int(g >= gmin), coll_ok=int(c >= cmin))
            table.append(r)
    cols = ["seed", "iteration", "env_steps", "density", "eval_split", "eval_layout_seed", "n", "GOAL", "COLLISION", "INFEASIBLE", "TIMEOUT", "stalled",
            "goal_rate", "goal_lo", "goal_hi", "coll_rate", "coll_lo", "coll_hi", "timeout_rate", "goal_ok", "coll_ok",
            "form_err_mean", "goal_steps_mean", "ckpt"]
    write_rows(out / "gate_by_checkpoint.csv", [{k: r[k] for k in cols} for r in table])

    pooled = [r for r in table if r["density"] == "pooled"]
    key = lambda r: (r["goal_rate"], r["iteration"])                      # pre-registered selection rule
    best = max(pooled, key=key)
    best_per_seed = {s: max([r for r in pooled if r["seed"] == s], key=key) for s in sorted({r["seed"] for r in pooled})}
    passes = lambda r: bool(r["goal_ok"] and r["coll_ok"])
    per_density = lambda r: {str(x["density"]): dict(goal_rate=x["goal_rate"], coll_rate=x["coll_rate"],
                                                     goal_ok=x["goal_ok"], coll_ok=x["coll_ok"])
                             for x in table if x["ckpt"] == r["ckpt"] and x["density"] != "pooled"}
    sel = dict(rule=cfg["gate.selection"], goal_rate_min=gmin, collision_rate_min=cmin,
               validation_trials_per_density=a.batch, env_fingerprint=cfg.env_fingerprint(),
               pending_placeholders=len(cfg.pending()), n_checkpoints=len(pooled),
               best=dict(ckpt=best["ckpt"], seed=best["seed"], iteration=best["iteration"], goal_rate=best["goal_rate"],
                         coll_rate=best["coll_rate"], gate_pass=passes(best), per_density=per_density(best)),
               best_per_seed={str(s): dict(ckpt=r["ckpt"], iteration=r["iteration"], goal_rate=r["goal_rate"],
                                           coll_rate=r["coll_rate"], gate_pass=passes(r), per_density=per_density(r))
                              for s, r in best_per_seed.items()},
               any_checkpoint_passes=any(passes(r) for r in pooled), wall_s=round(time.time() - t0, 1))
    (out / "selection.json").write_text(json.dumps(sel, indent=1))
    print(json.dumps(sel, indent=1))
    write_markdown(out / "GATE.md", table, sel, cfg)


def write_markdown(path, table, sel, cfg):
    pct = lambda r, k: f"{100 * r[k + '_rate']:.1f} [{100 * r[k + '_lo']:.1f}, {100 * r[k + '_hi']:.1f}]"
    L = ["# Gate: filter-OFF baseline per checkpoint", "",
         f"Validation layouts, {sel['validation_trials_per_density']} trials per density, VmasEnv, deterministic policy "
         f"(mean action); layout seeds `validation_base + density` (sample_layout, batch = trials per density). "
         f"Rates in % with Wilson 95% CIs. Thresholds (pre-registered, pending approval): pooled GOAL "
         f">= {100 * sel['goal_rate_min']:.0f}%, pooled COLLISION >= {100 * sel['collision_rate_min']:.0f}%. "
         f"Env fingerprint `{sel['env_fingerprint']}`, {sel['pending_placeholders']} config values pending approval.", "",
         f"**Any checkpoint passes: {'YES' if sel['any_checkpoint_passes'] else 'NO'}.** "
         f"Best by the selection rule: `{sel['best']['ckpt']}` "
         f"(GOAL {100 * sel['best']['goal_rate']:.1f}%, COLLISION {100 * sel['best']['coll_rate']:.1f}%, "
         f"gate {'PASS' if sel['best']['gate_pass'] else 'FAIL'}).", "",
         "| seed | iter | env steps | density | GOAL % | COLLISION % | TIMEOUT % | stalled | goal ok | coll ok |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        L.append(f"| {r['seed']} | {r['iteration']} | {r['env_steps']} | {r['density']} | {pct(r, 'goal')} | "
                 f"{pct(r, 'coll')} | {100 * r['timeout_rate']:.1f} | {r['stalled']} | {r['goal_ok']} | {r['coll_ok']} |")
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
