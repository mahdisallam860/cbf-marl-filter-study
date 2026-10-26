#!/usr/bin/env python3
"""EXPLORATORY, POST HOC: clearance thresholds of the filter-OFF final-test runs (K = 1, 2, 3),
read from the saved per-step logs (results/final_test/steps/*_off.npz) and saved layouts; no
simulation. Clearances are CONTROL-INSTANT SAMPLED MINIMA over the executed states t = 0..T
(initial and terminal included), not continuous-path minima:
    robot-obstacle: axle-centre distance - (R + R_o);   robot-robot: centre distance - 2R.
For x in X:
  (a) retention   = #(off GOAL and min clearance >= x) / #(off GOAL)
  (b) matched     = #(off GOAL and min clearance >= x) / #(all trials), next to the filter-on GOAL
                    rate on the same layouts.
Optional: AUC of the filter-off minimum obstacle clearance for predicting 'filter on did not reach
GOAL' among filter-off GOAL trials (descriptive; AUC = P(clearance of a not-GOAL case < clearance of a
GOAL case), ties counted 1/2).
Writes results/final_test/mechanism/clearance_thresholds*.csv and the MECHANISM_EXPLORATORY.md section."""
import csv, gzip, json, sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cbf_study.config import StudyConfig      # noqa: E402
from cbf_study.evaluate import write_rows     # noqa: E402

FT = ROOT / "results" / "final_test"
OUT = FT / "mechanism"
MD = FT / "MECHANISM_EXPLORATORY.md"
X = (0.0, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10)
LABEL = "EXPLORATORY, POST HOC"
BEGIN, END = "<!-- BEGIN clearance-threshold section -->", "<!-- END clearance-threshold section -->"


def auc(score_pos, score_neg):
    """P(score_pos < score_neg) + 0.5 P(=): 'positive' = filter on not GOAL, lower clearance = higher risk."""
    s = np.r_[score_pos, score_neg]
    r = rankdata(-s)                                   # high rank = low clearance
    n1, n0 = len(score_pos), len(score_neg)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")


def main():
    cfg = StudyConfig.load()
    R, Ro = cfg.protocol_config().robot_r, cfg.protocol_config().obst_r
    sel = json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())
    pols = [p["policy_id"] for p in sel["policies"]]
    oc = {}
    for p in sorted((FT / "cells").glob("*.csv.gz")):
        for r in csv.DictReader(gzip.open(p, "rt")):
            oc[(r["policy_id"], int(r["density"]), int(r["trial"]), r["condition"])] = r["outcome"]
    trials = []
    i, j = np.triu_indices(3, 1)
    for pid in pols:
        for K in (1, 2, 3):
            obst = np.load(FT / "layouts" / f"{pid}_K{K}.npz")["obst_xy"]
            z = dict(np.load(FT / "steps" / f"{pid}_K{K}_off.npz"))
            T, po, P = z["executed_periods"], z["pose_offsets"], z["poses"].astype(np.float64)
            for t in range(len(T)):
                S = P[po[t]:po[t] + T[t] + 1]
                go = (np.linalg.norm(S[:, :, None, :2] - obst[t][None, None], axis=3) - R - Ro).min()
                gr = (np.linalg.norm(S[:, i, :2] - S[:, j, :2], axis=2) - 2 * R).min()
                trials.append(dict(policy_id=pid, density=K, trial=t, off=oc[(pid, K, t, "off")], on=oc[(pid, K, t, "on")],
                                   min_gap_obst=float(go), min_gap_rr=float(gr)))
    write_rows(OUT / "clearance_thresholds_trials.csv", [dict(label=LABEL, **r) for r in trials])
    scopes = [("pooled", "all", lambda r: True)] + [("density", str(K), (lambda K: lambda r: r["density"] == K)(K)) for K in (1, 2, 3)] \
        + [("actor", p, (lambda p: lambda r: r["policy_id"] == p)(p)) for p in pols]
    rows, aucs = [], []
    for sc, sid, f in scopes:
        rs = [r for r in trials if f(r)]
        n, og = len(rs), [r for r in rs if r["off"] == "GOAL"]
        on_goal = sum(r["on"] == "GOAL" for r in rs)
        for kind in ("obst", "rr"):
            for x in X:
                k = sum(r[f"min_gap_{kind}"] >= x for r in og)
                rows.append(dict(label=LABEL, scope=sc, scope_id=sid, clearance=kind, x_m=x, n_trials=n, n_off_goal=len(og),
                                 n_off_goal_clear=k, retention=k / len(og), matched_rate=k / n, off_goal_rate=len(og) / n,
                                 on_goal_rate_same_layouts=on_goal / n, n_on_goal=on_goal))
        pos = np.array([r["min_gap_obst"] for r in og if r["on"] != "GOAL"])
        neg = np.array([r["min_gap_obst"] for r in og if r["on"] == "GOAL"])
        aucs.append(dict(label=LABEL, scope=sc, scope_id=sid, n_off_goal=len(og), n_on_not_goal=len(pos), n_on_goal=len(neg),
                         auc_min_obstacle_clearance=auc(pos, neg)))
    write_rows(OUT / "clearance_thresholds.csv", rows)
    write_rows(OUT / "clearance_thresholds_auc.csv", aucs)
    write_md(rows, aucs)


def write_md(rows, aucs):
    get = lambda sc, sid, kind: [r for r in rows if r["scope"] == sc and r["scope_id"] == sid and r["clearance"] == kind]
    L = [BEGIN, "", "## 5. Clearance thresholds of the filter-off runs (EXPLORATORY, POST HOC)", "",
         "**EXPLORATORY, POST HOC.** From the saved filter-off final-test trajectories (K = 1, 2, 3, 27,000 trials), "
         "control-instant SAMPLED minima over the executed states (initial and terminal included): robot-obstacle clearance = "
         "axle-centre distance - (R + R_o), robot-robot = centre distance - 2R. Generated by `scripts/clearance_thresholds.py`; "
         "all scopes (pooled, by density, by actor) with denominators in `mechanism/clearance_thresholds.csv`.", "",
         "This describes how close the unfiltered runs came to obstacles and to each other. It does not measure what a buffer or "
         "any filter would cost: a run that keeps a larger clearance is a different run, not this run with its close passes "
         "removed. Similar aggregate numbers do not establish an inevitable cost or hardware unacceptability. The 3.3 cm median "
         "of section 4 applies to group A only.", "",
         "(a) retention = share of filter-off GOAL trials whose sampled minimum clearance is >= x; (b) matched-denominator rate = "
         "filter-off GOAL trials with clearance >= x divided by ALL trials, shown next to the filter-on GOAL rate on the same "
         "layouts.", ""]
    for kind, title in (("obst", "Robot-obstacle clearance"), ("rr", "Robot-robot clearance")):
        L += [f"### {title}", "", "| scope | n trials | off GOAL | filter-on GOAL rate | " + " | ".join(f"x = {x:g} m: (a) / (b)" for x in X) + " |",
              "|---|---|---|---|" + "---|" * len(X)]
        for sc, sid in [("pooled", "all")] + [("density", k) for k in ("1", "2", "3")]:
            g = get(sc, sid, kind)
            L.append(f"| {sid if sc != 'pooled' else 'pooled'}{' (K)' if sc == 'density' else ''} | {g[0]['n_trials']} | {g[0]['n_off_goal']} "
                     f"({100 * g[0]['off_goal_rate']:.1f}%) | {100 * g[0]['on_goal_rate_same_layouts']:.1f}% | "
                     + " | ".join(f"{100 * r['retention']:.1f}% / {100 * r['matched_rate']:.1f}%" for r in g) + " |")
        L.append("")
    L += ["### Optional: AUC of the filter-off minimum obstacle clearance (descriptive)", "",
          "Among filter-off GOAL trials, how well a smaller filter-off minimum obstacle clearance separates pairs where filter on "
          "did not reach GOAL from pairs where it did (0.5 = no separation). Descriptive only; selected by observed outcomes.", "",
          "| scope | off GOAL trials | on not GOAL | on GOAL | AUC |", "|---|---|---|---|---|"]
    for a in aucs:
        if a["scope"] in ("pooled", "density"):
            L.append(f"| {a['scope_id'] if a['scope'] == 'density' else 'pooled'}{' (K)' if a['scope'] == 'density' else ''} | {a['n_off_goal']} | "
                     f"{a['n_on_not_goal']} | {a['n_on_goal']} | {a['auc_min_obstacle_clearance']:.3f} |")
    L += ["", "Per-actor AUCs: `mechanism/clearance_thresholds_auc.csv`.", "", END]
    text = MD.read_text()
    if BEGIN in text:
        text = text[:text.index(BEGIN)] + "\n".join(L) + text[text.index(END) + len(END):]
    else:
        text = text.rstrip("\n") + "\n\n" + "\n".join(L) + "\n"
    MD.write_text(text)


if __name__ == "__main__":
    main()
