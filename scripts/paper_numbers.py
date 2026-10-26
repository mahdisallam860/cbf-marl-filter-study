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
    revision_macros(mac)
    out = ["% Generated by scripts/paper_numbers.py from committed results; do not edit by hand.",
           "% Anchor check passed: confirmatory macros identical from verdict.json and endpoints.csv."]
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(mac.items())]
    (ROOT / "paper").mkdir(exist_ok=True)
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(out) + "\n")
    print(f"anchor check passed; {len(mac)} macros written")


def revision_macros(mac):
    """Macros added in the paper revision (results/paper_revision/, gate selections, MAPPO check,
    engineering stress summary, final-test cells for the H3 breakdown)."""
    import gzip
    R = ROOT / "results" / "paper_revision"
    c = lambda x: f"{int(x):,}".replace(",", "{,}")
    p1 = lambda x: num(100 * float(x), 1)
    # (a) time to goal
    ttg = list(csv.DictReader(open(R / "time_to_goal.csv")))
    rows = []
    for arm, an in (("on", "On"), ("bmatch", "Bm")):
        for K, kn in KN.items():
            r = next(x for x in ttg if x["test_arm"] == arm and x["scope"] == "density" and x["scope_id"] == str(K))
            mac[f"TtgPairs{an}{kn}"] = c(r["joint_success_pairs"])
            mac[f"TtgEq{an}{kn}"] = num(float(r["equal_actor_mean_diff"]), 2) if r["equal_actor_mean_diff"] not in ("", "nan") else "n/a"
            mac[f"TtgMed{an}{kn}"] = num(float(r["pooled_median_diff_pairweighted"]), 1)
            mac[f"TtgHl{an}{kn}"] = num(float(r["hl_pairweighted"]), 1)
            mac[f"TtgP{an}{kn}"] = sci(float(r["wilcoxon_p_pairweighted"]))
            mac[f"TtgSlower{an}{kn}"] = p1(r["share_slower"])
            per = [int(v.split(":")[1]) for v in r["joint_success_per_actor"].split(";")]
            mac[f"TtgMinActor{an}{kn}"] = c(min(per))
            rows.append(f"{'filter on' if arm == 'on' else 'B-match'} & {K} & {mac[f'TtgPairs{an}{kn}']} & {mac[f'TtgMinActor{an}{kn}']} & "
                        f"{mac[f'TtgEq{an}{kn}']} & {mac[f'TtgMed{an}{kn}']} & {mac[f'TtgHl{an}{kn}']} & {mac[f'TtgSlower{an}{kn}']} & {mac[f'TtgP{an}{kn}']}\\\\")
    mac["TabTtgRows"] = "\n".join(rows)
    # (b) per-seed table rows
    ps = list(csv.DictReader(open(R / "per_seed_endpoints.csv")))
    mac["TabPerSeedRows"] = "\n".join(
        f"{r['seed'].replace('seed', '')} & {r['density']} & {p1(r['fail_off'])} $\\to$ {p1(r['fail_on'])} & {num(100 * float(r['fail_diff']), 1)} "
        f"[{num(100 * float(r['fail_lo']), 1)}, {num(100 * float(r['fail_hi']), 1)}] & {num(100 * float(r['goal_loss']), 1)} "
        f"[{num(100 * float(r['goal_lo']), 1)}, {num(100 * float(r['goal_hi']), 1)}] & {num(float(r['form_inc']), 4)} "
        f"[{num(float(r['form_lo']), 4)}, {num(float(r['form_hi']), 4)}]\\\\" for r in sorted(ps, key=lambda r: (r['density'], r['seed'])))
    # (c) validation vs test
    vt = list(csv.DictReader(open(R / "validation_vs_test.csv")))
    mac["TabValTestRows"] = "\n".join(
        f"{r['density']} & {'validation' if r['source'].startswith('validation') else 'final test'} & {c(r['n_pairs'])} & {p1(r['fail_off'])} $\\to$ {p1(r['fail_on'])} & "
        f"{p1(r['goal_off'])} $\\to$ {p1(r['goal_on'])} & {num(100 * float(r['goal_loss']), 1)} [{num(100 * float(r['goal_lo']), 1)}, {num(100 * float(r['goal_hi']), 1)}] & "
        f"{num(float(r['form_inc']), 4)} [{num(float(r['form_lo']), 4)}, {num(float(r['form_hi']), 4)}]\\\\" for r in vt)
    # (d) stress cells
    st = list(csv.DictReader(open(R / "stress_cells.csv")))
    mac["TabStressRows"] = "\n".join(f"{r['density']} & {r['arm']} & {c(r['n'])} & {p1(r['GOAL'])} & {p1(r['COLLISION'])} & {p1(r['INFEASIBLE'])} & {p1(r['TIMEOUT'])}\\\\" for r in st)
    # (e) baseline (per actor, GOAL / COLLISION / TIMEOUT %)
    bl = list(csv.DictReader(open(R / "filter_off_baseline.csv")))
    Ks = (1, 2, 3, 4, 5, 8)
    lines = []
    for pid in [p["policy_id"] for p in json.loads((ROOT / "results/phase1/checkpoint_selection.json").read_text())["policies"]]:
        cells_ = []
        for K in Ks:
            r = next(x for x in bl if x["policy_id"] == pid and int(x["density"]) == K)
            from decimal import Decimal, ROUND_HALF_EVEN
            pc0 = lambda x: str((Decimal(x) * 100).quantize(Decimal(1), rounding=ROUND_HALF_EVEN))   # exact decimal rounding of the CSV text
            cells_.append(f"{pc0(r['GOAL'])}/{pc0(r['COLLISION'])}/{pc0(r['TIMEOUT'])}")
        lines.append(pid.replace("_", "\\_") + " & " + " & ".join(cells_) + "\\\\")
    mac["TabBaselineRows"] = "\n".join(lines)
    # (f) example selection
    ex = json.loads((R / "example_selection.json").read_text())
    mac.update(ExActor=ex["policy_id"].replace("_", "\\_"), ExK=str(ex["density"]), ExTrial=str(ex["trial"]),
               ExGapCm=num(100 * ex["selected_min_gap_obst_m"], 2), ExMedCm=num(100 * ex["group_A_median_m"], 2),
               ExCaState=str(ex["off_closest_approach_state"]), ExStallState=str(ex["on_stall_onset_state"]),
               ExOffT=str(ex["off_executed_periods"]), ExCoverH=num(ex["cover_disc_robot_h_m2"], 4),
               ExCoverGapCm=num(100 * ex["cover_disc_robot_body_clearance_m"], 1), ExRobot=str(ex["cover_disc_robot"]))
    # gate history and MAPPO check
    mac["GateVoneBest"] = p1(json.loads((ROOT / "results/gate/selection.json").read_text())["best"]["goal_rate"])
    mac["GateVtwoBest"] = p1(json.loads((ROOT / "results/gate_v2/selection.json").read_text())["best"]["goal_rate"])
    ev = list(csv.DictReader(open(ROOT / "results/mappo_validation/eval.csv")))
    last = max((r for r in ev if r["policy"] == "deterministic"), key=lambda r: int(r["iteration"]))
    mac["MappoNavSuccess"] = p1(last["success"])
    mac["MappoNavRandom"] = p1(next(r for r in ev if r["policy"] == "uniform_random")["success"])
    # engineering stress results that existed before the lock (stand-in random / attack controllers)
    ss = list(csv.DictReader(open(ROOT / "results/stress_summary.csv")))
    mac.update(EngStressTrials=c(sum(int(r["trials"]) for r in ss)), EngStressOffColl=c(sum(int(r["off_collision"]) for r in ss)),
               EngStressOnColl=c(sum(int(r["on_collision"]) for r in ss)), EngStressOnInf=c(sum(int(r["on_infeasible"]) for r in ss)))
    # H3 breakdown of filter-on non-GOAL trials (final test cells; stalled = protocol flag)
    tot = dict(n=0, nongoal=0, coll=0, inf=0, to_st=0, to_mv=0)
    for K, kn in KN.items():
        d = dict(n=0, nongoal=0, coll=0, inf=0, to_st=0, to_mv=0)
        for p in sorted((FT / "cells").glob(f"*_K{K}.csv.gz")):
            for r in csv.DictReader(gzip.open(p, "rt")):
                if r["condition"] != "on":
                    continue
                d["n"] += 1
                o = r["outcome"]
                if o == "GOAL":
                    continue
                d["nongoal"] += 1
                d["coll"] += o == "COLLISION"; d["inf"] += o == "INFEASIBLE"
                if o == "TIMEOUT":
                    d["to_st" if r["stalled"] == "1" else "to_mv"] += 1
        for k in tot:
            tot[k] += d[k]
        mac.update({f"HthreeNongoal{kn}": c(d["nongoal"]), f"HthreeInf{kn}": c(d["inf"]), f"HthreeStalled{kn}": c(d["to_st"]),
                    f"HthreeMoving{kn}": c(d["to_mv"]), f"HthreeShare{kn}": num(100 * (d["inf"] + d["to_st"]) / d["nongoal"], 1)})
    mac.update(HthreeNongoalAll=c(tot["nongoal"]), HthreeShareAll=num(100 * (tot["inf"] + tot["to_st"]) / tot["nongoal"], 1),
               HthreeMovingAll=c(tot["to_mv"]))


if __name__ == "__main__":
    main()
