#!/usr/bin/env python3
"""Phase 1: geometry-only solvability audit (same planner and settings as scripts/audit_solvability.py)
of the validation layouts at K = 1, 3 (now candidate study densities; the earlier audit covered 2, 5, 8)
and K = 4 (intermediate diagnostic; seed validation_base + 4 has not been used before). Validation split
only; no pilot or test layouts. Writes results/phase1/solvability/{layouts.csv, summary.csv, SOLVABILITY.md}."""
import json, sys, time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import audit_solvability as audit      # noqa: E402  (same audit_chunk as the earlier audit)

DENSITIES = (1, 3, 4)
N = 200


def main():
    from cbf_study.config import StudyConfig
    from cbf_study.evaluate import write_rows
    from cbf_study.solvability import BLOCKED, FOUND, UNDETERMINED, AuditConfig
    from cbf_study.stats import wilson_ci
    cfg = StudyConfig.load()
    ac = AuditConfig.from_study(cfg)
    out = ROOT / "results" / "phase1" / "solvability"
    out.mkdir(parents=True, exist_ok=True)
    tasks = [("validation", K, N, lo, min(N, lo + 25), None) for K in DENSITIES for lo in range(0, N, 25)]
    t0 = time.time()
    with ProcessPoolExecutor(4) as ex:
        rows = [r for chunk in ex.map(audit.audit_chunk, tasks) for r in chunk]
    write_rows(out / "layouts.csv", rows)
    summ = []
    for K in DENSITIES:
        rs = [r for r in rows if r["density"] == K]
        s = dict(split="validation", density=K, layout_seed=rs[0]["layout_seed"], n=len(rs),
                 role="intermediate diagnostic" if K == 4 else "candidate study density")
        for q in ("a", "b", "c"):
            st = [r[f"{q}_status"] for r in rs]
            for lab in (FOUND, BLOCKED, UNDETERMINED):
                s[f"{q}_{lab}"] = st.count(lab)
            lo, hi = wilson_ci(s[f"{q}_{FOUND}"], len(rs))
            s[f"{q}_found_lo"], s[f"{q}_found_hi"] = lo, hi
            key = "a_steps_max" if q == "a" else f"{q}_steps"
            steps = sorted(r[key] for r in rs if r[f"{q}_status"] == FOUND)
            s[f"{q}_steps_median"] = steps[len(steps) // 2] if steps else None
            s[f"{q}_steps_max"] = steps[-1] if steps else None
        summ.append(s)
    write_rows(out / "summary.csv", summ)
    meta = dict(audit_config=asdict(ac), env_fingerprint=cfg.env_fingerprint(), wall_s=round(time.time() - t0, 1))
    (out / "audit_config.json").write_text(json.dumps(meta, indent=1))
    L = ["# Phase 1: solvability of the validation layouts at K = 1, 3, 4", "",
         "Same planner, settings and status rules as `results/solvability/` (geometry only; FOUND = verified "
         f"collision-free plan within {ac.max_steps} steps; BLOCKED = proven disconnected; UNDETERMINED = neither). "
         "(a) each robot alone, (b) the team, (c) the team within the formation tolerance during transit. "
         "K = 4 is an intermediate diagnostic density, not a study density.", "",
         "| K | role | layouts (seed) | (a) found / blocked / undet. | (b) found / blocked / undet. | (c) found / blocked / undet. | (b) steps median / max | (c) found, Wilson 95% CI |",
         "|---|---|---|---|---|---|---|---|"]
    for s in summ:
        t = lambda q: f"{s[q + '_found']} / {s[q + '_blocked']} / {s[q + '_undetermined']}"
        L.append(f"| {s['density']} | {s['role']} | {s['n']} ({s['layout_seed']}) | {t('a')} | {t('b')} | {t('c')} | "
                 f"{s['b_steps_median']} / {s['b_steps_max']} | {100 * s['c_found_lo']:.1f}-{100 * s['c_found_hi']:.1f}% |")
    (out / "SOLVABILITY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), f"\nwall {meta['wall_s']} s")


if __name__ == "__main__":
    main()
