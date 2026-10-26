#!/usr/bin/env python3
"""Paper figures (vector PDF) from committed result files only; no simulation.
F1 validation sweep: filter-on GOAL and INFEASIBLE per gain, K = 1-3 pooled (results/validation_sweep/selection.csv)
F2 final test: outcome shares per density for filter off / filter on / B-match (results/final_test/endpoints.csv)
F3 final test: per-actor GOAL loss (off - on) with 95% Newcombe intervals, grouped by seed (endpoints.csv)
F4 exploratory: filter-off minimum obstacle clearance, group A vs group B (mechanism/clearance_trajectories.csv)
F5 exploratory: matched-denominator rate vs clearance threshold, with the filter-on GOAL rate (mechanism/clearance_thresholds.csv)
Writes paper/figures/F*.pdf."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "results" / "final_test"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "figure.dpi": 100})
C = {"off": "#4c72b0", "on": "#dd8452", "bmatch": "#55a868", "GOAL": "#4c72b0", "COLLISION": "#c44e52",
     "INFEASIBLE": "#8172b3", "TIMEOUT": "#937860"}
END = list(csv.DictReader(open(FT / "endpoints.csv")))


def f1():
    rows = list(csv.DictReader(open(ROOT / "results/validation_sweep/selection.csv")))
    a = [float(r["alpha"]) for r in rows]
    n = np.array([int(r["n"]) for r in rows])
    g = np.array([int(r["GOAL"]) for r in rows]) / n
    i = np.array([int(r["INFEASIBLE"]) for r in rows]) / n
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    x = np.arange(len(a))
    ax.bar(x - 0.18, 100 * g, 0.36, label="GOAL", color=C["GOAL"])
    ax.bar(x + 0.18, 100 * i, 0.36, label="INFEASIBLE", color=C["INFEASIBLE"])
    ax.set_xticks(x, [f"{v:g}" for v in a]); ax.set_xlabel(r"barrier gain $\alpha$ [1/s]")
    ax.set_ylabel("% of filter-on trials"); ax.legend(frameon=False)
    ax.set_title("Validation sweep, K = 1-3 pooled (development data)", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "F1_validation_sweep.pdf"); plt.close(fig)


def pooled(K, ref, test):
    return next(e for e in END if e["scope"] == "pooled" and e["density"] == str(K) and e["reference"] == ref and e["test"] == test)


def f2():
    fig, ax = plt.subplots(figsize=(6.8, 2.5))
    labels, pos = [], []
    for k, K in enumerate((1, 2, 3)):
        for j, (arm, row, side) in enumerate((("filter off", pooled(K, "off", "on"), "ref"), ("filter on", pooled(K, "off", "on"), "test"),
                                              ("B-match", pooled(K, "off", "bmatch"), "test"))):
            n = int(row["n_pairs"]); x = 4 * k + j; bottom = 0
            for o in ("GOAL", "COLLISION", "INFEASIBLE", "TIMEOUT"):
                v = 100 * int(row[f"{side}_{o}"]) / n
                ax.bar(x, v, 0.8, bottom=bottom, color=C[o], label=o if (k, j) == (0, 0) else None); bottom += v
            labels.append(arm); pos.append(x)
        ax.text(4 * k + 1, 103, f"K = {K}", ha="center")
    ax.set_xticks(pos, labels, rotation=30, ha="right"); ax.set_ylim(0, 112); ax.set_ylabel("% of 9,000 trials")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.30))
    fig.tight_layout(); fig.savefig(OUT / "F2_final_outcomes.pdf"); plt.close(fig)


def f3():
    fig, axs = plt.subplots(1, 3, figsize=(6.8, 2.4), sharey=True)
    for ax, K in zip(axs, (1, 2, 3)):
        rs = [e for e in END if e["scope"] == "policy" and e["density"] == str(K) and e["reference"] == "off" and e["test"] == "on"]
        for i, r in enumerate(rs):
            s = int(r["scope_id"][1])
            l, lo, hi = 100 * float(r["b_loss"]), 100 * float(r["b_ci95_lo"]), 100 * float(r["b_ci95_hi"])
            ax.errorbar(i + s, l, yerr=[[l - lo], [hi - l]], fmt="o", ms=3, color=["#4c72b0", "#dd8452", "#55a868"][s], capsize=2)
        ax.axhline(5, ls="--", lw=0.8, color="k"); ax.axhline(0, lw=0.5, color="grey")
        ax.set_xticks([i + int(r["scope_id"][1]) for i, r in enumerate(rs)], [r["scope_id"].replace("_", " ") for r in rs], rotation=60, fontsize=6)
        ax.set_title(f"K = {K}", fontsize=8)
    axs[0].set_ylabel("GOAL loss off - on [pp], 95% CI")
    fig.tight_layout(); fig.savefig(OUT / "F3_per_actor_goal_loss.pdf"); plt.close(fig)


def f4():
    rows = list(csv.DictReader(open(FT / "mechanism/clearance_trajectories.csv")))
    a = np.array([float(r["min_gap_obst"]) for r in rows if r["in_A"] == "1"])
    b = np.array([float(r["min_gap_obst"]) for r in rows if r["group"] == "B" and np.isfinite(float(r["min_gap_obst"]))])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for v, lab, c in ((a, f"group A (n = {len(a):,})", C["on"]), (b, f"group B (n = {len(b):,})", C["off"])):
        x = np.sort(100 * v); ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=c, label=lab)
    ax.set_xlabel("filter-off min obstacle clearance [cm] (sampled)"); ax.set_ylabel("ECDF"); ax.set_xlim(0, 40)
    ax.legend(frameon=False, fontsize=7); ax.set_title("Exploratory, post hoc", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "F4_clearance_groups.pdf"); plt.close(fig)


def f5():
    rows = list(csv.DictReader(open(FT / "mechanism/clearance_thresholds.csv")))
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for sid, c in (("1", "#4c72b0"), ("2", "#dd8452"), ("3", "#55a868")):
        rs = [r for r in rows if r["scope"] == "density" and r["scope_id"] == sid and r["clearance"] == "obst"]
        x = [100 * float(r["x_m"]) for r in rs]; y = [100 * float(r["matched_rate"]) for r in rs]
        ax.plot(x, y, "o-", ms=3, color=c, label=f"K = {sid}")
        ax.axhline(100 * float(rs[0]["on_goal_rate_same_layouts"]), ls="--", lw=0.8, color=c)
    ax.set_xlabel("obstacle clearance threshold x [cm]"); ax.set_ylabel("% of all trials")
    ax.set_title("off GOAL with clearance >= x (solid); filter-on GOAL (dashed)", fontsize=7)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "F5_matched_denominator.pdf"); plt.close(fig)


for fn in (f1, f2, f3, f4, f5):
    fn()
print("figures written:", sorted(p.name for p in OUT.glob("*.pdf")))
