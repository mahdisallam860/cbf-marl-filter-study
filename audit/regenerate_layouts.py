#!/usr/bin/env python3
"""Provenance audit, part 2: regenerate every recorded validation and pilot layout set at the
sampler commit each run was launched from and at the current commit, in read-only git worktrees,
under the pinned environment (this repository's venv), and compare the arrays byte for byte.

Reads only: git objects, results/, the pre-change hash record. Writes only under audit/ (and the
worktrees, made read-only). Launch commits come from output birth times vs commit times
(audit/PROVENANCE.md). No layouts were saved by the original runs, so regeneration can show that
the code at those commits produces these arrays now; it cannot prove the arrays the runs used then.

Outputs: audit/layout_hashes_by_commit.csv, audit/recorded_layout_ids.csv,
audit/layout_outcome_crosscheck.csv, audit/layout_audit_summary.json,
audit/layouts_HEAD/*.npz (arrays at the current commit, for future comparisons)."""
import csv, glob, hashlib, json, os, subprocess, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WT_BASE = Path("/home/user/audit_worktrees")
SCRATCH = Path("/home/user/audit_regen_out")      # non-HEAD arrays (not committed)
RUN_AS = ["runuser", "-u", "nobody", "--"]          # unprivileged: the read-only bits actually bind
PY = str(ROOT / "venv" / "bin" / "python")

# run -> commit whose code it ran (evidence in audit/PROVENANCE.md)
RUNS = {
    "train_v1_phase1": "871206b", "train_v1_phase2": "a99fcc2", "gate_v1": "1280a8a",
    "solvability_audit": "ba87dbf", "curriculum_launch": "5b2820f", "curriculum_seed1_resume": "d112bc3",
    "gate_v2": "ff46268",
}
HEAD = "ff46268"


def sh(*a, **k):
    return subprocess.run(a, check=True, capture_output=True, text=True, cwd=ROOT, **k).stdout.strip()


def worktree(c):
    d = WT_BASE / c
    if not d.exists():
        sh("git", "worktree", "add", "--detach", str(d), c)
        subprocess.run(["chmod", "-R", "a-w", str(d)], check=True)
    for f in (d, d / "cbf_study", d / "cbf_study" / "scenario.py", d / "config" / "study.toml"):
        assert subprocess.run(RUN_AS + ["test", "-w", str(f)]).returncode != 0, f"{f} is writable by the child"
    return d


def regenerate(c):
    d = worktree(c)
    out = (ROOT / "audit" / "layouts_HEAD") if c == HEAD else (SCRATCH / c)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chown", "nobody", str(out)], check=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    r = subprocess.run(RUN_AS + [PY, "-I", str(ROOT / "audit" / "_regen_child.py"), str(d), str(out)],
                       check=True, capture_output=True, text=True, env=env)
    subprocess.run(["chown", "-R", "root", str(out)], check=True)
    info = json.loads(r.stdout)
    info["outdir"] = str(out)
    info["blobs"] = {f: sh("git", "rev-parse", f"{c}:{f}")[:12] for f in
                     ("cbf_study/scenario.py", "cbf_study/cbf_qp.py", "cbf_study/config.py", "config/study.toml")}
    return info


def load(outdir, split, K, n):
    z = np.load(Path(outdir) / f"{split}_K{K}_n{n}.npz")
    return {k: z[k] for k in z.files}


def identical(a, b):
    return all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and a[k].tobytes() == b[k].tobytes() for k in a) \
        and set(a) == set(b)


def recorded_ids():
    """(run, run_seed, split, K, layout_seed, n, source, recorded_at) for every evaluation output."""
    ids = []
    for path, run in (("results/gate/gate_by_checkpoint.csv", "gate_v1"), ("results/gate_v2/gate_by_checkpoint.csv", "gate_v2")):
        for r in csv.DictReader(open(ROOT / path)):
            if r["density"] == "pooled":
                continue
            when = "added after the run (formula validation_base+K)" if run == "gate_v1" else "written by the run"
            ids.append((run, int(r["seed"]), r["eval_split"], int(r["density"]), int(r["eval_layout_seed"]), int(r["n"]), path, when))
    for p in sorted(glob.glob(str(ROOT / "results/curriculum/seed_*/heldout.csv"))):
        s = int(Path(p).parent.name.split("_")[1])
        for r in csv.DictReader(open(p)):
            rec = (s == 1 and int(r["iteration"]) <= 546)
            run = "curriculum_seed1_resume" if (s == 1 and not rec and int(r["iteration"]) > 546) else "curriculum_launch"
            when = "reconstructed (re-evaluated checkpoint)" if rec else "written by the run"
            ids.append((run, s, "validation", int(r["density"]), int(r["layout_seed"]), int(r["n"]),
                        str(Path(p).relative_to(ROOT)), when))
    for r in csv.DictReader(open(ROOT / "results/solvability/layouts.csv")):
        ids.append(("solvability_audit", -1, r["split"], int(r["density"]), int(r["layout_seed"]),
                    200 if r["split"] == "validation" else 500, "results/solvability/layouts.csv", "written by the run"))
    return ids


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    subprocess.run(["chmod", "755", str(SCRATCH)], check=True)
    infos = {c: regenerate(c) for c in sorted(set(RUNS.values()))}
    pinned = json.loads(open(ROOT / "audit" / "recorded_hashes_pre_v1.1.json").read())
    rows, diffs = [], []
    for c, info in infos.items():
        for r in info["rows"]:
            a = load(info["outdir"], r["split"], r["density"], r["n_layouts"])
            b = load(infos[HEAD]["outdir"], r["split"], r["density"], r["n_layouts"])
            same = identical(a, b)
            key = f"{r['split']}_K{r['density']}_n{r['n_layouts']}"
            rec = pinned["hashes"].get(key)
            row = dict(commit=c, runs="+".join(k for k, v in RUNS.items() if v == c), **{f"blob_{Path(f).name}": h for f, h in info["blobs"].items()},
                       config_version=info["config_version"], env_fingerprint=info["env_fingerprint"], numpy=info["numpy"],
                       split=r["split"], density=r["density"], layout_seed=r["layout_seed"], n_layouts=r["n_layouts"],
                       sha256_init_poses=r["sha256_init_poses"], sha256_obst_xy=r["sha256_obst_xy"],
                       sha256_goal=r["sha256_goal"], sha256_slots=r["sha256_slots"], dtypes_shapes=r["dtypes_shapes"],
                       byte_identical_to_HEAD=int(same), recorded_hash_pre_v1_1=rec or "",
                       recorded_hash_match="" if rec is None else int(rec == r["sha256_16_init_obst"]))
            rows.append(row)
            if not same or (rec is not None and rec != r["sha256_16_init_obst"]):
                diffs.append(row)
    out = ROOT / "audit"
    with open(out / "layout_hashes_by_commit.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # every layout ID a run recorded, checked against the regenerated set at its launch commit and at HEAD
    by = {(r["commit"], r["split"], r["density"]): r for r in rows}
    agg = {}
    for run, s, split, K, seed, n, src, when in recorded_ids():
        k = (run, s, split, K, seed, n, src, when)
        agg[k] = agg.get(k, 0) + 1
    rid = []
    for (run, s, split, K, seed, n, src, when), cnt in sorted(agg.items(), key=lambda x: (x[0][0], x[0][1], x[0][2], x[0][3])):
        c = RUNS[run]
        g = by.get((c, split, K))
        rid.append(dict(run=run, run_seed=s if s >= 0 else "", split=split, density=K, layout_seed=seed, n_layouts=n,
                        rows_using_it=cnt, source=src, recorded=when, launch_commit=c,
                        seed_matches_formula=int(g is not None and g["layout_seed"] == seed),
                        n_matches=int(g is not None and g["n_layouts"] == n),
                        sha256_init_poses_at_launch=g["sha256_init_poses"] if g else "",
                        sha256_obst_xy_at_launch=g["sha256_obst_xy"] if g else "",
                        byte_identical_launch_vs_HEAD=g["byte_identical_to_HEAD"] if g else ""))
    with open(out / "recorded_layout_ids.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rid[0])); w.writeheader(); w.writerows(rid)

    # outcome-level cross-check: same final-stage checkpoints, curriculum held-out (sampler v1 for seeds
    # 0 and 2 at run time) vs gate_v2 (sampler v1.1 now); identical layouts must give identical counts
    gv2 = {(r["ckpt"], r["density"]): r for r in csv.DictReader(open(ROOT / "results/gate_v2/gate_by_checkpoint.csv"))}
    xc = []
    for p in sorted(glob.glob(str(ROOT / "results/curriculum/seed_*/heldout.csv"))):
        for r in csv.DictReader(open(p)):
            if r["stage"] != "4":
                continue
            ck = r["ckpt"].replace("stage_checkpoints", "checkpoints")
            ck = str(Path(ck).parent / ("ckpt_it" + ck.split("_it")[-1]))
            g = gv2.get((ck, r["density"]))
            if g is None:
                continue
            n = int(r["n"])
            same = all(abs(float(r[a]) * n - int(g[b])) < 1e-6 for a, b in (("goal", "GOAL"), ("collision", "COLLISION"), ("timeout", "TIMEOUT")))
            xc.append(dict(ckpt=ck, density=r["density"], heldout_goal=r["goal"], heldout_collision=r["collision"],
                           heldout_timeout=r["timeout"], gate_v2_GOAL=g["GOAL"], gate_v2_COLLISION=g["COLLISION"],
                           gate_v2_TIMEOUT=g["TIMEOUT"], n=n, identical=int(same)))
    with open(out / "layout_outcome_crosscheck.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(xc[0])); w.writeheader(); w.writerows(xc)

    summ = dict(commits={c: dict(runs=[k for k, v in RUNS.items() if v == c], blobs=i["blobs"], config_version=i["config_version"],
                                 env_fingerprint=i["env_fingerprint"], numpy=i["numpy"], scenario_file=i["scenario_file"])
                         for c, i in infos.items()},
                layout_sets_per_commit=len(infos[HEAD]["rows"]), comparisons=len(rows), differences=len(diffs),
                recorded_ids=len(rid), recorded_ids_seed_or_n_mismatch=sum(1 for r in rid if not (r["seed_matches_formula"] and r["n_matches"])),
                recorded_ids_not_identical_to_HEAD=sum(1 for r in rid if r["byte_identical_launch_vs_HEAD"] != 1),
                pinned_hashes_checked=sum(1 for r in rows if r["recorded_hash_pre_v1_1"]),
                pinned_hash_mismatches=sum(1 for r in rows if r["recorded_hash_match"] == 0),
                outcome_crosscheck_rows=len(xc), outcome_crosscheck_mismatches=sum(1 for r in xc if not r["identical"]),
                original_layouts_saved_by_runs=False)
    (out / "layout_audit_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "commits"}, indent=1))
    for c, i in summ["commits"].items():
        print(c, i["runs"], i["blobs"]["cbf_study/scenario.py"], i["blobs"]["config/study.toml"], "v", i["config_version"], i["env_fingerprint"])
    if diffs:
        print("DIFFERENCES:", json.dumps(diffs[:10], indent=1))
        sys.exit(1)


if __name__ == "__main__":
    main()
