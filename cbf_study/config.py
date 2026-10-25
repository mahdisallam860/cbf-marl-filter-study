"""
Single study configuration (config/study.toml) for training, evaluation and the filter.

Every study value in the file is {value, approval, note}; "pending" marks a placeholder pending
advisor approval. This module builds the FilterConfig, ProtocolConfig and ScenarioConfig that the
rest of the code uses, so no study path relies on dataclass defaults or hard-coded numbers.

check_paired() is the guard that filter-on and filter-off trials use identical values: the two
ProtocolConfigs must be equal and the two FilterConfigs may differ only in alpha (the barrier gain,
which only the filter uses).

env_fingerprint() hashes every value that defines the environment a policy acts in (dynamics,
radii, bounds, layouts, scoring, observation). A checkpoint stores it; a policy is only evaluated
under the same fingerprint.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass, fields, replace
from pathlib import Path

from .cbf_qp import FilterConfig
from .protocol import ProtocolConfig, check_consistent
from .scenario import ScenarioConfig

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "config" / "study.toml"
STUDY_SECTIONS = ("timing", "robot", "obstacles", "scenario", "filter", "protocol", "reward",
                  "observation", "gate", "seeds")
# sections / keys that define the environment a policy sees (training == evaluation)
ENV_KEYS = ("timing", "robot", "obstacles", "scenario", "filter.ell", "protocol", "observation")
APPROVAL_STATES = ("pending", "approved")


@dataclass(frozen=True)
class ObsConfig:
    n_agents: int
    robot_r: float
    obst_r: float
    k_obst: int
    obs_range: float
    goal_clip: float


@dataclass(frozen=True)
class RewardConfig:
    w_progress: float
    w_formation: float
    w_collision_obst: float
    w_collision_rr: float
    w_control: float


class StudyConfig:
    def __init__(self, raw: dict, path: str = "<dict>"):
        self.path = str(path)
        self.values: dict[str, dict] = {}
        self.approval: dict[str, str] = {}
        for sec in STUDY_SECTIONS:
            if sec not in raw:
                raise ValueError(f"{path}: missing section [{sec}]")
            self.values[sec] = {}
            for key, ent in raw[sec].items():
                if not (isinstance(ent, dict) and "value" in ent and "approval" in ent):
                    raise ValueError(f"{path}: [{sec}].{key} must be {{value, approval, note}}")
                if ent["approval"] not in APPROVAL_STATES:
                    raise ValueError(f"{path}: [{sec}].{key} approval must be one of {APPROVAL_STATES}")
                self.values[sec][key] = ent["value"]
                self.approval[f"{sec}.{key}"] = ent["approval"]
        self.mappo = dict(raw.get("mappo", {}))
        self._validate()

    @classmethod
    def load(cls, path=DEFAULT_PATH):
        with open(path, "rb") as f:
            return cls(tomllib.load(f), path)

    def __getitem__(self, dotted):
        sec, key = dotted.split(".")
        return self.values[sec][key]

    def pending(self):
        return sorted(k for k, a in self.approval.items() if a == "pending")

    # ---- dataclasses used by the code ---------------------------------------------------------
    @property
    def dt(self):
        return float(self["timing.dt"])

    @property
    def densities(self):
        return tuple(int(k) for k in self["obstacles.densities"])

    @property
    def alphas(self):
        return tuple(float(a) for a in self["filter.alpha_sweep"])

    def filter_config(self, alpha) -> FilterConfig:
        v = self.values
        fc = FilterConfig(dt=self.dt, robot_r=float(v["robot"]["robot_r"]), obst_r=float(v["obstacles"]["obst_r"]),
                          ell=float(v["filter"]["ell"]), v_min=float(v["robot"]["v_min"]),
                          v_max=float(v["robot"]["v_max"]), w_max=float(v["robot"]["w_max"]),
                          alpha=float(alpha), sensing_range=float(v["filter"]["sensing_range"]),
                          comm_range=float(v["filter"]["comm_range"]),
                          intervention_tol=float(v["filter"]["intervention_tol"]),
                          solver_tol=float(v["filter"]["solver_tol"]))
        fc.validate()
        return fc

    def protocol_config(self) -> ProtocolConfig:
        v = self.values
        return ProtocolConfig(dt=self.dt, robot_r=float(v["robot"]["robot_r"]), obst_r=float(v["obstacles"]["obst_r"]),
                              v_min=float(v["robot"]["v_min"]), v_max=float(v["robot"]["v_max"]),
                              w_max=float(v["robot"]["w_max"]), max_steps=int(v["timing"]["max_steps"]),
                              goal_tol=float(v["protocol"]["goal_tol"]),
                              formation_tol=float(v["protocol"]["formation_tol"]),
                              substeps=int(v["protocol"]["substeps"]),
                              stall_window=int(v["protocol"]["stall_window"]),
                              stall_dist=float(v["protocol"]["stall_dist"]))

    def scenario_config(self) -> ScenarioConfig:
        s = self.values["scenario"]
        return ScenarioConfig(n_agents=int(self["robot.n_agents"]), slot_radius=float(s["slot_radius"]),
                              start_centroid=tuple(map(float, s["start_centroid"])),
                              goal_centroid=tuple(map(float, s["goal_centroid"])),
                              obst_x=tuple(map(float, s["obst_x"])), obst_y=tuple(map(float, s["obst_y"])),
                              obst_gap=float(s["obst_gap"]), edge_clear=float(s["edge_clear"]),
                              start_xy_std=float(s["start_xy_std"]), start_th_std=float(s["start_th_std"]))

    def layout_filter_config(self) -> FilterConfig:
        """FilterConfig handed to scenario.sample_layout (it reads ell, radii and the derived cover
        distances only; alpha does not affect layouts, the first sweep value is used)."""
        return self.filter_config(self.alphas[0])

    def obs_config(self) -> ObsConfig:
        o = self.values["observation"]
        return ObsConfig(n_agents=int(self["robot.n_agents"]), robot_r=float(self["robot.robot_r"]),
                         obst_r=float(self["obstacles.obst_r"]), k_obst=int(o["k_obst"]),
                         obs_range=float(o["obs_range"]), goal_clip=float(o["goal_clip"]))

    def reward_config(self) -> RewardConfig:
        return RewardConfig(**{f.name: float(self.values["reward"][f.name]) for f in fields(RewardConfig)})

    # ---- seeds ----------------------------------------------------------------------------------
    def train_seed(self, run_seed, density, chunk):
        if not (0 <= run_seed < 10 and 0 <= density < 100 and 0 <= chunk < 1000):
            raise ValueError("train seed components out of range")
        return int(self["seeds.train_base"]) + run_seed * 100000 + density * 1000 + chunk

    def validation_seed(self, density):
        return int(self["seeds.validation_base"]) + density

    def pilot_seed(self, density):
        return int(self["seeds.pilot_base"]) + density

    # ---- fingerprints -----------------------------------------------------------------------
    def env_values(self):
        out = {}
        for k in ENV_KEYS:
            if "." in k:
                out[k] = self[k]
            else:
                out[k] = self.values[k]
        return out

    def env_fingerprint(self):
        blob = json.dumps(self.env_values(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    def as_dict(self):
        return dict(values=self.values, approval=self.approval, mappo=self.mappo)

    # ---- validation ---------------------------------------------------------------------------
    def _validate(self):
        pc, sc = self.protocol_config(), self.scenario_config()
        for a in self.alphas:
            fc = self.filter_config(a)          # validate(): alpha*dt <= 1, ell > 0, ranges, box
            check_consistent(pc, fc)
        if sc.n_agents != self.obs_config().n_agents:
            raise ValueError("n_agents mismatch")
        bases = sorted(int(self[f"seeds.{k}"]) for k in ("train_base", "validation_base", "pilot_base"))
        if bases[1] - bases[0] < 10 * 100000 or bases[2] - bases[1] < 10 * 100000:
            raise ValueError("seed bases must be at least 1e6 apart so train/validation/pilot layouts never coincide")
        if not self.densities or min(self.densities) < 0:
            raise ValueError("densities must be non-negative obstacle counts")


def check_paired(pc_off: ProtocolConfig, fc_off: FilterConfig, pc_on: ProtocolConfig, fc_on: FilterConfig):
    """Filter-on and filter-off must use identical values: same ProtocolConfig, and FilterConfigs
    that differ in alpha only. Raises ValueError otherwise."""
    if pc_off != pc_on:
        raise ValueError(f"filter-off and filter-on ProtocolConfig differ: {pc_off} vs {pc_on}")
    if replace(fc_off, alpha=fc_on.alpha) != fc_on:
        raise ValueError(f"filter-off and filter-on FilterConfig differ beyond alpha: {fc_off} vs {fc_on}")
    check_consistent(pc_off, fc_off)
    check_consistent(pc_on, fc_on)
