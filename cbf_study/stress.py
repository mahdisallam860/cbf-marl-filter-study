"""Adversarial stress runs for the filter. Evidence only: zero violations here is consistent with
the one-period lemma in docs/CBF_DESIGN.md, it does not prove continuous-time safety."""
from __future__ import annotations

import numpy as np

from .protocol import Outcome, run_paired
from .scenario import sample_layout


class AttackPolicy:
    """Deterministic: each robot drives at the nearest obstacle centre or neighbour at full speed."""

    def __init__(self, fc, k_w=3.0):
        self.fc, self.k_w = fc, k_w

    def __call__(self, poses, layout):
        B, N, _ = poses.shape
        xy = poses[..., :2]
        targets = np.concatenate([np.broadcast_to(layout.obst_xy[:, None], (B, N) + layout.obst_xy.shape[1:]),
                                  np.broadcast_to(xy[:, None], (B, N, N, 2))], axis=2)   # [B,N,K+N,2]
        d = np.linalg.norm(targets - xy[:, :, None], axis=3)
        d[:, np.arange(N), layout.obst_xy.shape[1] + np.arange(N)] = np.inf
        tgt = targets[np.arange(B)[:, None], np.arange(N)[None], d.argmin(axis=2)]
        e = tgt - xy
        err = (np.arctan2(e[..., 1], e[..., 0]) - poses[..., 2] + np.pi) % (2 * np.pi) - np.pi
        return np.stack([np.full((B, N), self.fc.v_max), np.clip(self.k_w * err, -self.fc.w_max, self.fc.w_max)], axis=2)


class RandomBoxPolicy:
    """Seeded uniform random commands in the velocity box (deterministic given the seed)."""

    def __init__(self, fc, seed):
        self.fc, self.rng = fc, np.random.default_rng(seed)

    def __call__(self, poses, layout):
        B, N, _ = poses.shape
        return np.stack([self.rng.uniform(self.fc.v_min, self.fc.v_max, (B, N)),
                         self.rng.uniform(-self.fc.w_max, self.fc.w_max, (B, N))], axis=2)


def run_stress(env_factory, pc, fc, n_obst, batch, seed, which):
    lay = sample_layout(seed, batch, n_obst, fc)
    pol = AttackPolicy(fc) if which == "attack" else RandomBoxPolicy(fc, seed)
    off, on = run_paired(env_factory, pol, lay, pc, fc, log_steps=False) if which == "attack" else (None, None)
    if which != "attack":      # random policy is stateful: build a fresh, identically seeded one per condition
        from .protocol import run_trials
        off = run_trials(env_factory(), RandomBoxPolicy(fc, seed), lay, pc, fc, False, log_steps=False)
        on = run_trials(env_factory(), RandomBoxPolicy(fc, seed), lay, pc, fc, True, log_steps=False)
    return off, on


def summarise(off, on):
    c = lambda r, o: int((r.outcome == o).sum())
    return dict(trials=len(on.outcome),
                off_collision=c(off, Outcome.COLLISION), on_collision=c(on, Outcome.COLLISION),
                on_infeasible=c(on, Outcome.INFEASIBLE), on_goal=c(on, Outcome.GOAL),
                on_timeout=c(on, Outcome.TIMEOUT), off_goal=c(off, Outcome.GOAL),
                on_h_obst_min=float(on.h_obst_min.min()), on_h_rr_min=float(on.h_rr_min.min()),
                on_interventions=int(on.n_interventions.sum()))
