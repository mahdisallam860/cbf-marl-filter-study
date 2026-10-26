#!/usr/bin/env python3
"""Writes paper/numbers.tex: every empirical number used in the paper draft, as LaTeX macros, read
from committed result files only (no simulation). Anchor check: the confirmatory macros are built
from results/final_test/verdict.json AND independently from results/final_test/endpoints.csv
(pooled filter-on vs filter-off rows); the formatted strings must be identical, otherwise the
script exits non-zero and writes nothing.

Sources: results/final_test/{verdict.json, endpoints.csv, cells/}, results/validation_sweep/selection.csv,
results/final_test/mechanism/{timeout_classes.csv, clearance_summary.csv, clearance_summary.json,
clearance_thresholds.csv, clearance_thresholds_auc.csv, closing_speed_vs_cap.csv},
results/final_test/infeasibility/{summary.json, infeasible_robots.csv, initial_state_negative_barriers.csv},
results/design/margin_constants.json, results/phase1/checkpoint_selection.json."""
import csv, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FT = ROOT / "results" / "final_test"
M = FT / "mechanism"
KN = {1: "one", 2: "two", 3: "three"}


def num(x, p):
    s = f"{x:.{p}f}"
    return f"\\ensuremath{{-{s[1:]}}}" if s.startswith("-") else s


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return f"\\ensuremath{{{m}\\times 10^{{{int(e)}}}}}"


def confirmatory_from(verdict=None, endpoints=None):
    out = {}
    for K, kn in KN.items():
        if verdict is not None:
            v = verdict["verdict_per_density"][str(K)]
            a, b, c = v["a"], v["b"], v["c"]
            vals = dict(adiff=a["diff_on_minus_off"], aup=a["upper_98_33"], alo=a["ci95"][0], ahi=a["ci95"][1], ap=a["p_one_sided"],
                        apass=a["passed"], bloss=b["goal_loss"], bup=b["upper_98_33"], blo=b["ci95"][0], bhi=b["ci95"][1], bpass=b["passed"],
                        cinc=c["increase_m"], cup=c["upper_98_33_bootstrap"], cnorm=c["normal_crosscheck"], clo=c["ci95"][0], chi=c["ci95"][1],
                        celig=c["eligible"], cexcl=c["excluded"], cpass=c["passed"], verdict=v["verdict"])
        else:
            r = next(e for e in endpoints if e["scope"] == "pooled" and e["density"] == str(K) and e["test"] == "on" and e["reference"] == "off")
            f = lambda k: float(r[k])
            vals = dict(adiff=f("a_diff"), aup=f("a_upper_9833"), alo=f("a_ci95_lo"), ahi=f("a_ci95_hi"), ap=f("a_p_one_sided"),
                        apass=r["a_meets"] == "1", bloss=f("b_loss"), bup=f("b_upper_9833"), blo=f("b_ci95_lo"), bhi=f("b_ci95_hi"),
                        bpass=r["b_meets"] == "1", cinc=f("c_estimate"), cup=f("c_upper_9833"), cnorm=f("c_normal_upper"),
                        clo=f("c_ci95_lo"), chi=f("c_ci95_hi"), celig=int(r["c_eligible"]), cexcl=int(r["c_excluded"]),
                        cpass=r["c_meets"] == "1", verdict=r["usefulness_dev"])
        P = lambda ok: "met" if ok else "not met"
        out.update({f"Adiff{kn}": num(100 * vals["adiff"], 2), f"Aupper{kn}": num(100 * vals["aup"], 2),
                    f"AciLo{kn}": num(100 * vals["alo"], 2), f"AciHi{kn}": num(100 * vals["ahi"], 2), f"Ap{kn}": sci(vals["ap"]) if vals["ap"] > 0 else "\\ensuremath{<10^{-300}}",
                    f"Apass{kn}": P(vals["apass"]), f"Bloss{kn}": num(100 * vals["bloss"], 2), f"Bupper{kn}": num(100 * vals["bup"], 2),
                    f"BciLo{kn}": num(100 * vals["blo"], 2), f"BciHi{kn}": num(100 * vals["bhi"], 2), f"Bpass{kn}": P(vals["bpass"]),
                    f"Cinc{kn}": num(vals["cinc"], 4), f"Cupper{kn}": num(vals["cup"], 4), f"Cnorm{kn}": num(vals["cnorm"], 4),
                    f"CciLo{kn}": num(vals["clo"], 4), f"CciHi{kn}": num(vals["chi"], 4), f"Celig{kn}": f"{vals['celig']:,}".replace(",", "{,}"),
                    f"Cexcl{kn}": str(vals["cexcl"]), f"Cpass{kn}": P(vals["cpass"]), f"Verdict{kn}": vals["verdict"]})
    return out


def main():
    verdict = json.loads((FT / "verdict.json").read_text())
    endpoints = list(csv.DictReader(open(FT / "endpoints.csv")))
    A = confirmatory_from(verdict=verdict)
    B = confirmatory_from(endpoints=endpoints)
    bad = {k: (A[k], B[k]) for k in A if A[k] != B[k]}
    if bad:
        sys.exit(f"ANCHOR CHECK FAILED: {bad}")
    mac = dict(A)
    pooled = lambda K, ref, test: next(e for e in endpoints if e["scope"] == "pooled" and e["density"] == str(K) and e["reference"] == ref and e["test"] == test)
    onbm = []
    for K, kn in KN.items():
        r, rb, rob = pooled(K, "off", "on"), pooled(K, "off", "bmatch"), pooled(K, "bmatch", "on")
        n = int(r["n_pairs"])
        pct = lambda x: num(100 * int(x) / n, 1)
        mac.update({f"GoalOff{kn}": pct(r["ref_GOAL"]), f"GoalOn{kn}": pct(r["test_GOAL"]), f"GoalBm{kn}": pct(rb["test_GOAL"]),
                    f"FailOff{kn}": num(100 * (int(r["ref_COLLISION"]) + int(r["ref_INFEASIBLE"])) / n, 1),
                    f"FailOn{kn}": num(100 * (int(r["test_COLLISION"]) + int(r["test_INFEASIBLE"])) / n, 1),
                    f"CollOff{kn}": f"{int(r['ref_COLLISION']):,}".replace(",", "{,}"), f"CollOn{kn}": r["test_COLLISION"],
                    f"InfOn{kn}": r["test_INFEASIBLE"], f"CollBm{kn}": rb["test_COLLISION"], f"InfBm{kn}": rb["test_INFEASIBLE"],
                    f"TOoff{kn}": pct(r["ref_TIMEOUT"]), f"TOon{kn}": pct(r["test_TIMEOUT"]), f"TObm{kn}": pct(rb["test_TIMEOUT"]),
                    f"OnMinusBmGoal{kn}": num(100 * (int(rob["test_GOAL"]) - int(rob["ref_GOAL"])) / n, 1),
                    f"RgrOff{kn}": pct(r["ref_reached_goal_region"]), f"RgrOn{kn}": pct(r["test_reached_goal_region"])})
        onbm.append(100 * (int(rob["test_GOAL"]) - int(rob["ref_GOAL"])) / n)
    mac["OnBmGoalMin"], mac["OnBmGoalMax"] = num(min(onbm), 1), num(max(onbm), 1)
    mac["PairsPerDensity"] = "9{,}000"
    ntr = sum(int(pooled(K, "off", "on")["n_pairs"]) for K in KN)
    mac["TrialsTotal"] = f"{ntr:,}".replace(",", "{,}")
    # validation sweep selection
    sel = list(csv.DictReader(open(ROOT / "results/validation_sweep/selection.csv")))
    an = {"0.5": "half", "1": "one", "2": "two", "5": "five", "10": "ten"}
    for r in sel:
        k = an[r["alpha"].rstrip("0").rstrip(".") if "." in r["alpha"] else r["alpha"]]
        mac[f"SelGoal{k}"] = f"{int(r['GOAL']):,}".replace(",", "{,}")
        mac[f"SelInf{k}"] = f"{int(r['INFEASIBLE']):,}".replace(",", "{,}")
    mac["SelN"] = f"{int(sel[0]['n']):,}".replace(",", "{,}")
    mac["AlphaStar"] = num(json.loads((ROOT / "results/validation_sweep/alpha_star.json").read_text())["alpha_star"], 0)
    # mechanism: timeout classes (filter on)
    tc = [r for r in csv.DictReader(open(M / "timeout_classes.csv")) if r["condition"] == "on"]
    fi = [r for r in tc if r["category"].startswith("filter-induced")]
    ng = [r for r in tc if r["category"] == "near-goal parking stall"]
    oc = Counter(r["off_outcome"] for r in fi)
    mac.update(TOonTotal=f"{len(tc):,}".replace(",", "{,}"), TOfi=f"{len(fi):,}".replace(",", "{,}"), TOfiPct=num(100 * len(fi) / len(tc), 0),
               TOng=f"{len(ng):,}".replace(",", "{,}"), TOngPct=num(100 * len(ng) / len(tc), 0),
               TOngOffTO=f"{sum(r['off_outcome'] == 'TIMEOUT' for r in ng):,}".replace(",", "{,}"),
               TOfiOffColl=f"{oc['COLLISION']:,}".replace(",", "{,}"), TOfiOffGoal=f"{oc['GOAL']:,}".replace(",", "{,}"),
               TOfiObst=f"{sum('obstacle' in r['category'] for r in fi):,}".replace(",", "{,}"))
    # groups and clearances
    cj = json.loads((M / "clearance_summary.json").read_text())["group_counts"]
    mac.update(GroupA=f"{cj['A']:,}".replace(",", "{,}"), GroupAobst=f"{cj['A_obst']:,}".replace(",", "{,}"), GroupAnb=str(cj["A_nb"]),
               GroupAprime=f"{cj[chr(65) + chr(39)]:,}".replace(",", "{,}"), GroupB=f"{cj['B']:,}".replace(",", "{,}"))
    cs = list(csv.DictReader(open(M / "clearance_summary.csv")))
    g = lambda grp, q, f: next(r for r in cs if r["group"] == grp and r["scope"] == "pooled" and r["quantity"] == q)[f]
    mac.update(GapAmedCm=num(100 * float(g("A", "min_gap_obst", "p50")), 1), GapBmedCm=num(100 * float(g("B", "min_gap_obst", "p50")), 1),
               ACobstViolPct=num(100 * float(g("A", "C_obst_viol_any", "share")), 0), BCobstViolPct=num(100 * float(g("B", "C_obst_viol_any", "share")), 0),
               ACrrViolPct=num(100 * float(g("A", "C_rr_viol_any", "share")), 0), BCrrViolPct=num(100 * float(g("B", "C_rr_viol_any", "share")), 0),
               AhoNegPct=num(100 * float(g("A", "h_obst_neg_any", "share")), 0),
               AfirstObstPeriod=num(float(g("A", "C_obst_first_period", "p50")), 0))
    # phase-1 thresholds and AUC
    th = list(csv.DictReader(open(M / "clearance_thresholds.csv")))
    t = lambda sid, kind, x, f: float(next(r for r in th if r["scope_id"] == sid and r["clearance"] == kind and float(r["x_m"]) == x)[f])
    mac.update(ThrRetObstFive=num(100 * t("all", "obst", 0.05, "retention"), 1), ThrMatchObstFive=num(100 * t("all", "obst", 0.05, "matched_rate"), 1),
               ThrOnGoal=num(100 * t("all", "obst", 0.05, "on_goal_rate_same_layouts"), 1), ThrOffGoal=num(100 * t("all", "obst", 0.0, "matched_rate"), 1))
    auc = next(r for r in csv.DictReader(open(M / "clearance_thresholds_auc.csv")) if r["scope"] == "pooled")
    mac["AucPooled"] = num(float(auc["auc_min_obstacle_clearance"]), 2)
    # transit cap
    cap = [r for r in csv.DictReader(open(M / "closing_speed_vs_cap.csv")) if r["condition"] == "on"]
    mac.update(CapBindMin=num(100 * min(float(r["share_binding"]) for r in cap), 0), CapBindMax=num(100 * max(float(r["share_binding"]) for r in cap), 0),
               CapMedMin=num(min(float(r["cap_median"]) for r in cap), 2), CapMedMax=num(max(float(r["cap_median"]) for r in cap), 2))
    mc = json.loads((ROOT / "results/design/margin_constants.json").read_text())
    mac["Vp"] = num(mc["V_p"], 2)
    # infeasibility audit and A5
    s = json.loads((FT / "infeasibility/summary.json").read_text())
    rr = [r for r in csv.DictReader(open(FT / "infeasibility/infeasible_robots.csv")) if r["reproduced"] == "1"]
    z = [float(r["binding_zero_slack_f64"]) for r in rr]
    hd = [abs(float(r["binding_h_minus_hstar"])) for r in rr]
    init = [r for r in csv.DictReader(open(FT / "infeasibility/initial_state_negative_barriers.csv")) if r.get("source", "").startswith("final")]
    mac.update(InfTotal=str(s["infeasible_trials"]), InfRepro=str(s["reproduced"]), InfNear=str(s["near_boundary_reproduced_robots"]),
               InfRobots=str(s["infeasible_robots"]), InfSlackMin=sci(min(z)), InfSlackMax=sci(max(z)), InfHdiffMax=sci(max(hd)),
               AfiveCount=str(len(init)), AfiveH=num(float(init[0]["min_h_obst"]), 5) if init else "-",
               AfiveGap=num(float(init[0]["min_body_gap_obst"]), 3) if init else "-")
    out = ["% Generated by scripts/paper_numbers.py from committed results; do not edit by hand.",
           "% Anchor check passed: confirmatory macros identical from verdict.json and endpoints.csv."]
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(mac.items())]
    (ROOT / "paper").mkdir(exist_ok=True)
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(out) + "\n")
    print(f"anchor check passed; {len(mac)} macros written")


if __name__ == "__main__":
    main()
