#!/usr/bin/env python3
"""Provenance audit, part 3: curriculum seed 1 (crashed in the final stage on a sampler stall at
global iteration 565, resumed from the state saved at iteration 546).

Checks
  A. curve rows reconstructed from train.log (iterations 1-546) against train.log AND the
     independent stdout copy of the original process (audit backup), env-step arithmetic for all rows;
  B. stage checkpoints: metadata vs curve rows; which files predate the crash;
  C. held-out rows re-evaluated from saved checkpoints (iterations <= 546): evaluated again now in
     fresh processes and compared field by field; goal/collision/timeout vs the original log lines;
  D. resume state: the original run's latest.pt (pre-patch copy in the backup) vs its iteration-546
     checkpoint; MappoLearner.load round trip (weights, Adam moments and steps, value normaliser,
     sampling-generator state); the patch that added stage summaries; the logged resume values;
  E. provenance labels: audit/seed1_provenance.csv (row -> original or reconstructed, and checks).
Read-only on results/ and the backup; writes audit/seed1_*.{csv,json}."""
import csv, json, os, re, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
RUN = ROOT / "results/curriculum/seed_1"
BACKUP = Path(os.environ["AUDIT_BACKUP"])
CUT = 546                   # last iteration whose state was saved before the crash
IT = re.compile(r"^stage (\d+) it (\d+) steps (\d+)/(\d+) total (\d+) goal ([\d.na]+) coll ([\d.na]+) timeout ([\d.na]+) kl ([\d.na]+)")
HO = re.compile(r"K(\d+): goal ([\d.]+) coll ([\d.]+) timeout ([\d.]+)")
PER_IT = 64 * 384


def log_lines(path, stop_at_resume=True):
    its, ho, g = [], [], 0
    for line in open(path):
        line = line.rstrip("\n")
        if stop_at_resume and line.startswith("resumed"):
            break
        m = IT.match(line)
        if m:
            g += 1
            its.append(dict(iteration=g, stage=int(m.group(1)), stage_it=int(m.group(2)), stage_env_steps=int(m.group(3)),
                            env_steps=int(m.group(5)), goal=m.group(6), coll=m.group(7), timeout=m.group(8), kl=m.group(9)))
        elif line.strip().startswith("held-out") and its:
            for k, a, b, c in HO.findall(line):
                ho.append(dict(iteration=its[-1]["iteration"], density=int(k), goal=a, coll=b, timeout=c))
    return its, ho


def reeval(args):
    ck, dens, n = args
    import torch
    torch.set_num_threads(1)
    from cbf_study.config import StudyConfig
    from train_curriculum import heldout
    return ck, heldout(StudyConfig.load(), ROOT / ck, dens, n)


def main():
    import torch
    out, res = ROOT / "audit", {}
    prov = []
    # ---- A. curve rows ------------------------------------------------------------------
    curve = list(csv.DictReader(open(RUN / "curve.csv")))
    log_its, log_ho = log_lines(RUN / "train.log")
    std_its, std_ho = log_lines(BACKUP / "session_logs/cur_seed1.out")          # original process stdout
    assert len(std_its) >= CUT and len(log_its) >= CUT
    rec = [r for r in curve if int(r["iteration"]) <= CUT]
    mism_log = mism_std = 0
    for r in rec:
        i = int(r["iteration"])
        for src, lines in (("log", log_its), ("std", std_its)):
            L = lines[i - 1]
            same = (L["iteration"] == i and L["stage"] == int(r["stage"]) and L["stage_env_steps"] == int(r["stage_env_steps"])
                    and L["env_steps"] == int(r["env_steps"]) and L["goal"] == r["goal_rate"] and L["coll"] == r["collision_rate"]
                    and L["timeout"] == r["timeout_rate"] and L["kl"] == r["approx_kl"])
            if not same:
                if src == "log": mism_log += 1
                else: mism_std += 1
    arith = sum(int(r["env_steps"]) != int(r["iteration"]) * PER_IT for r in curve)
    contiguous = [int(r["iteration"]) for r in curve] == list(range(1, len(curve) + 1))
    discarded = [L for L in std_its if L["iteration"] > CUT]
    res["A_curve"] = dict(rows=len(curve), reconstructed_rows=len(rec), original_rows=len(curve) - len(rec),
                          reconstructed_vs_train_log_mismatches=mism_log, reconstructed_vs_original_stdout_mismatches=mism_std,
                          env_steps_not_equal_iteration_times_24576=arith, iterations_contiguous=contiguous,
                          original_iterations_after_last_save_discarded=[discarded[0]["iteration"], discarded[-1]["iteration"]] if discarded else [],
                          discarded_env_steps=len(discarded) * PER_IT)
    for r in curve:
        prov.append(dict(file="results/curriculum/seed_1/curve.csv", iteration=r["iteration"], density="",
                         provenance="reconstructed_from_train_log" if int(r["iteration"]) <= CUT else "original_run_output",
                         checked="matches train.log and original stdout" if int(r["iteration"]) <= CUT else "written by the resumed run"))

    # ---- B. stage checkpoints -----------------------------------------------------------------
    crash_t = os.stat(BACKUP / "session_logs/cur_seed1.out").st_mtime          # last write of the original process
    ck_rows, ck_bad = [], 0
    by_it = {int(r["iteration"]): r for r in curve}
    for p in sorted((RUN / "stage_checkpoints").glob("*.pt")):
        m = torch.load(p, map_location="cpu", weights_only=False)
        it = int(m["iteration"])
        ok = int(by_it[it]["env_steps"]) == int(m["env_steps"]) and int(by_it[it]["stage"]) == int(m["stage"])
        ck_bad += not ok
        ck_rows.append(dict(file=str(p.relative_to(ROOT)), iteration=it, env_steps=m["env_steps"], stage=m["stage"],
                            consistent_with_curve=int(ok), written_before_crash=int(os.stat(p).st_mtime < crash_t),
                            origin="original run" if it <= CUT else "resumed run"))
    res["B_checkpoints"] = dict(stage_checkpoints=len(ck_rows), inconsistent=ck_bad,
                                written_before_crash=sum(r["written_before_crash"] for r in ck_rows),
                                pre_crash_iterations_max=max(r["iteration"] for r in ck_rows if r["written_before_crash"]),
                                origin_vs_time_consistent=all((r["iteration"] <= CUT) == bool(r["written_before_crash"]) for r in ck_rows))
    with open(out / "seed1_checkpoints.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ck_rows[0])); w.writeheader(); w.writerows(ck_rows)

    # ---- C. held-out rows ---------------------------------------------------------------------
    ho = list(csv.DictReader(open(RUN / "heldout.csv")))
    rec_ho = [r for r in ho if int(r["iteration"]) <= CUT]
    jobs = {}
    for r in rec_ho:
        jobs.setdefault(r["ckpt"], []).append(int(r["density"]))
    with ProcessPoolExecutor(4) as ex:
        fresh = dict(ex.map(reeval, [(ck, d, int(rec_ho[0]["n"])) for ck, d in jobs.items()]))
    fields = ["goal", "collision", "timeout", "coll_obst", "coll_rr", "stalled", "timeout_goal_dist_median", "form_err_mean", "n", "layout_seed"]
    mism_fresh = mism_logged = 0
    lg = {(h["iteration"], h["density"]): h for h in std_ho}
    for r in rec_ho:
        e = fresh[r["ckpt"]][int(r["density"])]
        same = all((str(e[k]) == r[k]) or (r[k] == "nan" and str(e[k]) == "nan") for k in fields)
        mism_fresh += not same
        L = lg.get((int(r["iteration"]), int(r["density"])))
        same_log = L is not None and all(abs(float(r[a]) - float(L[b])) < 5e-4 for a, b in (("goal", "goal"), ("collision", "coll"), ("timeout", "timeout")))
        mism_logged += not same_log
        prov.append(dict(file="results/curriculum/seed_1/heldout.csv", iteration=r["iteration"], density=r["density"],
                         provenance="reconstructed_reevaluated_saved_checkpoint",
                         checked=f"re-evaluated again: {'identical' if same else 'DIFFERENT'}; vs original log: {'match' if same_log else 'MISMATCH'}"))
    for r in ho:
        if int(r["iteration"]) > CUT:
            prov.append(dict(file="results/curriculum/seed_1/heldout.csv", iteration=r["iteration"], density=r["density"],
                             provenance="original_run_output", checked="written by the resumed run"))
    res["C_heldout"] = dict(rows=len(ho), reconstructed_rows=len(rec_ho), original_rows=len(ho) - len(rec_ho),
                            reevaluated_again_mismatches=mism_fresh, vs_original_logged_values_mismatches=mism_logged,
                            logged_heldout_values_up_to_cut=sum(1 for h in std_ho if h["iteration"] <= CUT))
    st_json = json.load(open(RUN / "stages.json"))
    for s in st_json["stages"]:
        prov.append(dict(file="results/curriculum/seed_1/stages.json", iteration="", density=f"stage {s['stage']}",
                         provenance="reconstructed_from_train_log" if s.get("source") == "train.log" else "original_run_output",
                         checked="stage summary"))

    # ---- D. resume state ------------------------------------------------------------------------
    from cbf_study.mappo import MappoLearner
    saved = torch.load(BACKUP / "session_logs/seed1_latest_backup.pt", map_location="cpu", weights_only=False)
    ck546 = torch.load(next((RUN / "stage_checkpoints").glob(f"stage4_*_it{CUT:04d}.pt")), map_location="cpu", weights_only=False)
    eq = lambda a, b: a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    actor_vs_ckpt = eq(saved["actor"], ck546["actor_state"])
    run_cfg = json.load(open(RUN / "run.json"))
    hp = run_cfg["hyperparams"]
    lrn = MappoLearner(ck546["obs_dim"], ck546["n_agents"], ck546["lo"], ck546["hi"], hp, 1)
    lrn.load(saved)
    got = lrn.state()

    def opt_eq(a, b):
        if a["param_groups"] != b["param_groups"] or a["state"].keys() != b["state"].keys():
            return False
        return all(eq(a["state"][k], b["state"][k]) for k in a["state"])
    load_roundtrip = dict(actor=eq(got["actor"], saved["actor"]), critic=eq(got["critic"], saved["critic"]),
                          actor_adam=opt_eq(got["opt_a"], saved["opt_a"]), critic_adam=opt_eq(got["opt_c"], saved["opt_c"]),
                          value_normaliser=got["vnorm"] == saved["vnorm"], sampling_generator=torch.equal(got["gen"], saved["gen"]))
    adam_steps = sorted({float(v["step"]) for v in saved["opt_a"]["state"].values()})
    tmp = ROOT / "audit" / ".patch_roundtrip.pt"
    patched = dict(saved, stages=[{"stage": 0}])
    torch.save(patched, tmp)
    back = torch.load(tmp, map_location="cpu", weights_only=False)
    tmp.unlink()
    patch_roundtrip = all(eq(back[k], saved[k]) for k in ("actor", "critic")) and opt_eq(back["opt_a"], saved["opt_a"]) \
        and opt_eq(back["opt_c"], saved["opt_c"]) and back["vnorm"] == saved["vnorm"] and torch.equal(back["gen"], saved["gen"]) \
        and all(back[k] == saved[k] for k in ("stage", "stage_env_steps", "env_steps", "iteration", "stream_idx"))
    resume_line = next(l for l in open(RUN / "train.log") if l.startswith("resumed"))
    logged = re.findall(r"\d+", resume_line)
    res["D_resume"] = dict(saved_by="original run at its iteration-546 evaluation (pre-patch copy in the backup)",
                           saved_position=dict(stage=saved["stage"], stage_env_steps=saved["stage_env_steps"], env_steps=saved["env_steps"],
                                               iteration=saved["iteration"], stream_idx=saved["stream_idx"]),
                           saved_actor_equals_checkpoint_it546=actor_vs_ckpt, load_roundtrip_bitwise=load_roundtrip,
                           adam_step_count=adam_steps, patch_preserves_state=patch_roundtrip,
                           resume_log_line=resume_line.strip(),
                           resume_log_matches_saved=[int(x) for x in logged] == [saved["stage"], saved["stage_env_steps"], saved["env_steps"], saved["iteration"]],
                           patched_file_as_loaded="not retained: overwritten by the resumed run's own saves",
                           not_restored=["VMAS environment states and the episodes in progress at iteration 546",
                                         "position inside the partially consumed training-layout chunk (resumed at the next chunk)",
                                         "global torch/numpy RNG (not used for action sampling or layouts)",
                                         "iterations 547-565 of the original run (computed, then lost in the crash)"])
    res["E_labels"] = dict(rows_labelled=len(prov), reconstructed=sum(p["provenance"].startswith("reconstructed") for p in prov))
    with open(out / "seed1_provenance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prov[0])); w.writeheader(); w.writerows(prov)
    (out / "seed1_verification.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
