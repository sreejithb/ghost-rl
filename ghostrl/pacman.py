"""Pac-Man planner (v0, not frozen): stay where ghosts cannot cut him off, eat dots.

Reads the simulator state directly and assumes nothing about how ghosts pick
moves, only how fast they could get anywhere (`dist_ghost`, speed 15/16).
"""

from __future__ import annotations

import numpy as np

from .maze import UNREACHABLE, Maze
from .sim import State


class Planner:
    def __init__(self, maze: Maze, margin: float = 1.0, horizon: int = 40,
                 eps: float = 0.0, territory_cap: int = 80, ghost_speed: float = 15 / 16,
                 pellet_bonus: float = 8.0, pellet_bonus_danger: float = 40.0, seed: int = 0):
        self.pellet_bonus, self.pellet_bonus_danger = pellet_bonus, pellet_bonus_danger
        self.m, self.margin, self.horizon, self.eps = maze, margin, horizon, eps
        self.cap, self.speed = territory_cap, ghost_speed
        self.rng = np.random.default_rng(seed)

    def ghost_arrival(self, s: State) -> np.ndarray:
        """B x T: earliest tick any ghost could stand on each tile."""
        d = self.m.dist_ghost[s.ghost_pos * 4 + s.ghost_dir].astype(np.float64)  # B x 4 x T
        d = np.where(d >= UNREACHABLE, np.inf, d / self.speed)
        # A frightened ghost is harmless until its fright runs out.
        d = np.where(s.frightened[..., None], np.maximum(d, s.fright_left[:, None, None]), d)
        return d.min(1)

    def act(self, s: State) -> np.ndarray:
        m = self.m
        B = s.B
        nxt = m.nbr[s.pac_pos]  # B x 4
        legal = nxt >= 0
        safe_nxt = np.maximum(nxt, 0)
        garr = self.ghost_arrival(s)  # B x T
        garr_r = np.repeat(garr, 4, axis=0)  # (B*4) x T
        start = nxt.reshape(-1)
        R = len(start)
        tp = np.full((R, m.T + 1), np.inf)  # last column is a sentinel for walls
        ok = (start >= 0) & (garr_r[np.arange(R), np.maximum(start, 0)] > 1)
        tp[np.flatnonzero(ok), start[ok]] = 1.0
        nb = np.where(m.nbr >= 0, m.nbr, m.T)  # T x 4
        limit = np.minimum(garr_r - self.margin, self.horizon + 1)  # R x T
        for _ in range(self.horizon):
            cand = tp[:, nb].min(-1) + 1  # R x T
            new = np.where(cand < limit, np.minimum(tp[:, :-1], cand), tp[:, :-1])
            if np.array_equal(new, tp[:, :-1]):
                break
            tp[:, :-1] = new
        tp = tp[:, :-1].reshape(B, 4, m.T)
        territory = np.isfinite(tp)  # B x 4 x T
        size = territory.sum(-1)
        ar = np.arange(B)[:, None]
        safe = legal & (garr[ar, safe_nxt] > 1 + self.margin * 0) & territory[
            np.arange(B)[:, None], np.arange(4)[None], safe_nxt]
        # nearest dot inside the territory (0 when none)
        # Power pellets pull harder than dots, much harder when a ghost is close.
        danger = garr[np.arange(B), s.pac_pos] < 12
        bonus = np.where(danger, self.pellet_bonus_danger, self.pellet_bonus)[:, None, None]
        cost = tp - np.where(m.power0, bonus, 0.0)
        dd = np.where(territory & s.dots[:, None, :], cost, np.inf).min(-1)
        dd = np.where(np.isfinite(dd), dd, 1e4)
        # Hunt frightened ghosts we can reach before their fright ends.
        hunt = np.zeros((B, 4))
        for g in range(4):
            dg = 1 + m.dist_free[safe_nxt, s.ghost_pos[:, g, None]].astype(np.float64)  # B x 4
            ok = s.frightened[:, g, None] & (dg + self.margin < s.fright_left[:, None]) & legal
            hunt = np.maximum(hunt, np.where(ok, 200.0 - dg, 0.0))
        score = np.minimum(size, self.cap) * 1000.0 - dd + hunt * 100.0
        # nothing safe: run for the tile ghosts reach last
        flee = np.where(legal, garr[ar, safe_nxt], -np.inf)
        score = np.where(safe.any(1, keepdims=True), np.where(safe, score, -np.inf), flee)
        score = score + self.rng.random(score.shape) * 1e-3
        a = score.argmax(1)
        if self.eps > 0:
            rnd = self.rng.random(B) < self.eps
            pick = np.where(np.isfinite(score), self.rng.random(score.shape), -1).argmax(1)
            a = np.where(rnd, pick, a)
        return a
