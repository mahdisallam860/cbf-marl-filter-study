"""
Frozen MAPPO policy for evaluation.

    pol = FrozenMAPPOPolicy("results/train/seed_0/checkpoints/ckpt_it0250.pt")
    u_nom = pol(poses [B,N,3], layout)        # -> [B,N,2] float64

The actor is loaded in eval mode with requires_grad=False and called under torch.inference_mode,
so evaluation cannot change it (param_digest() lets a run prove that). The action is the
deterministic one, the Gaussian mean, clipped to the velocity box; the observation is built by the
same observations.build_obs used in training. The filter never touches the policy: it receives
this action as u_nom.

A checkpoint is only accepted under the environment it was trained in: its env fingerprint
(dynamics, radii, bounds, layouts, scoring, observation) must equal that of the current
config/study.toml.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

from .config import StudyConfig
from .mappo import load_actor_checkpoint
from .observations import build_obs, obs_dim


class FrozenMAPPOPolicy:
    def __init__(self, ckpt_path, cfg: StudyConfig | None = None):
        cfg = StudyConfig.load() if cfg is None else cfg
        actor, meta = load_actor_checkpoint(ckpt_path)
        if meta["env_fingerprint"] != cfg.env_fingerprint():
            raise ValueError(f"{ckpt_path}: trained under env fingerprint {meta['env_fingerprint']}, "
                             f"current config has {cfg.env_fingerprint()}")
        self.oc = cfg.obs_config()
        if meta["obs_dim"] != obs_dim(self.oc):
            raise ValueError("observation size differs from the checkpoint")
        fc = cfg.layout_filter_config()
        self.lo = np.array([fc.v_min, -fc.w_max])
        self.hi = np.array([fc.v_max, fc.w_max])
        if not (np.allclose(meta["lo"], self.lo) and np.allclose(meta["hi"], self.hi)):
            raise ValueError("velocity box differs from the checkpoint")
        actor.eval()
        for p in actor.parameters():
            p.requires_grad_(False)
        self.actor = actor
        self.meta = {k: v for k, v in meta.items() if k not in ("actor_state", "config")}
        self.path = str(ckpt_path)
        self.digest = self.param_digest()

    def __call__(self, poses, layout):
        obs = build_obs(poses, layout, self.oc)
        with torch.inference_mode():
            mean = self.actor.mean(torch.from_numpy(obs))
        return np.clip(mean.double().numpy(), self.lo, self.hi)

    def param_digest(self):
        h = hashlib.sha256()
        for k, v in sorted(self.actor.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def assert_unchanged(self):
        if self.param_digest() != self.digest:
            raise RuntimeError(f"{self.path}: policy parameters changed during evaluation")
