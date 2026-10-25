"""The reserved final-test layout seeds (docs/final_test_seed_reservation.json) are disjoint from every
seed range the study has used, and no saved result records one. No layout is generated here."""
import csv
import json
import re
from pathlib import Path

from cbf_study.config import StudyConfig

ROOT = Path(__file__).resolve().parents[1]
RES = json.loads((ROOT / "docs" / "final_test_seed_reservation.json").read_text())
CFG = StudyConfig.load()


def test_reserved_seeds_unique_and_inside_the_reserved_range():
    lo, hi = RES["reserved_range"]
    seeds = [s["seed"] for s in RES["seeds"]]
    assert len(seeds) == len(set(seeds))
    assert all(lo <= s <= hi for s in seeds)
    assert all(s["seed"] == 4_000_000 + 1000 * s["policy_index"] + s["density"] for s in RES["seeds"])


def test_reserved_range_disjoint_from_train_validation_and_pilot_seeds():
    lo, hi = RES["reserved_range"]
    train = (CFG.train_seed(0, 0, 0), CFG.train_seed(9, 99, 999))
    assert train[1] < lo
    for K in range(100):
        for s in (CFG.validation_seed(K), CFG.pilot_seed(K)):
            assert not lo <= s <= hi


def test_no_saved_result_or_audit_record_uses_a_reserved_seed():
    lo, hi = RES["reserved_range"]
    hits = []
    for base in ("results", "audit"):
        for p in (ROOT / base).rglob("*.csv"):
            with open(p, newline="") as f:
                rd = csv.DictReader(f)
                cols = [c for c in (rd.fieldnames or []) if "seed" in c.lower()]
                for row in rd:
                    for c in cols:
                        for tok in re.findall(r"\d+", row[c] or ""):
                            if lo <= int(tok) <= hi:
                                hits.append((str(p), c, tok))
        for p in (ROOT / base).rglob("*.json"):
            for tok in re.findall(r'"[a-z_]*seed[a-z_]*":\s*"?(\d+)', p.read_text()):
                if lo <= int(tok) <= hi:
                    hits.append((str(p), "json", tok))
    assert not hits, hits[:5]
