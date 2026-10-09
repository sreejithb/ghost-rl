"""Pac-Man planner (v0, not frozen): stay where ghosts cannot cut him off, eat dots.

Reads the simulator state directly and assumes nothing about how ghosts pick
moves, only how fast they could get anywhere (`dist_ghost` and the known skip tick).
"""

from __future__ import annotations

import numpy as np

from .maze import REVERSE, UNREACHABLE, Maze
from .sim import State


class Planner:
    def __init__(self, maze: Maze, margin: float = 1.0, horizon: int = 40,
                 eps: float = 0.0, territory_cap: int = 80,
                 pellet_bonus: float = 8.0, pellet_bonus_danger: float = 40.0, seed: int = 0):
        self.pellet_bonus, self.pellet_bonus_danger = pellet_bonus, pellet_bonus_danger
        self.m, self.margin, self.horizon, self.eps = maze, margin, horizon, eps
        self.cap = territory_cap
        self.rng = np.random.default_rng(seed)

    def ghost_arrival(self, s: State) -> np.ndarray:
        """B x T: earliest tick any ghost could stand on each tile."""
        d = self.m.dist_ghost[s.ghost_pos * 4 + s.ghost_dir].astype(np.float64)  # B x 4 x T
        # Ghosts skip every 16th tick, and which tick that is is known: turn moves into ticks exactly.
        r = (s.tick % 16)[:, None, None]
        avail = np.where(r < 15, 15 - r, 0)  # moves possible before the next skip
        d = np.where(d >= UNREACHABLE, np.inf, d + np.where(d > avail, 1 + (d - avail - 1) // 15, 0))
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
        self.last_safe = safe.any(1)  # diagnostics: was any move safe this tick
        self.last_size = size
        score = np.where(safe.any(1, keepdims=True), np.where(safe, score, -np.inf), flee)
        score = score + self.rng.random(score.shape) * 1e-3
        a = score.argmax(1)
        if self.eps > 0:
            rnd = self.rng.random(B) < self.eps
            pick = np.where(np.isfinite(score), self.rng.random(score.shape), -1).argmax(1)
            a = np.where(rnd, pick, a)
        return a


class SearchPlanner:
    """Flat Monte-Carlo search: try each first move, play the game forward, keep the best average.

    Rollouts use the real simulator, so speeds, skip ticks, mode switches, pellets and the scripted
    frightened walk are exact. The only guess is what chasing ghosts do: they are modelled as
    path-chasers (each takes the legal move closest to Pac-Man by path length), a neutral worst case
    that favours neither team. Pac-Man's own rollout moves come from a cheap greedy rule.
    """

    def __init__(self, sim, n_rollouts: int = 12, depth: int = 25, seed: int = 0,
                 death_cost: float = 60.0, ghost_bonus: float = 6.0, dot_pull: float = 0.05,
                 eps: float = 0.0, reserve: bool = False, crn: bool = True,
                 heading_bonus: float = 0.3, reserve_range: int = 10):
        self.reserve, self.crn, self.heading_bonus, self.reserve_range = reserve, crn, heading_bonus, reserve_range
        self.sim, self.m = sim, sim.maze
        self.n, self.depth, self.rng = n_rollouts, depth, np.random.default_rng(seed)
        self.death_cost, self.ghost_bonus, self.dot_pull, self.eps = death_cost, ghost_bonus, dot_pull, eps

    # --- pieces of the rollout -------------------------------------------------

    def _ghost_chase(self, s):
        """B x 4 path-chase moves (legal, no reversing)."""
        m = self.m
        legal = self.sim.ghost_legal(s)  # B x 4 x 4
        nxt = np.maximum(m.nbr[s.ghost_pos], 0)  # B x 4 x 4
        d = m.dist_free[nxt, s.pac_pos[:, None, None]].astype(np.float64)
        return np.where(legal, d, np.inf).argmin(-1)

    def _hold_pellets(self, s):
        """R bool: keep the power pellets in reserve (no live ghost close, small dots still left)."""
        m = self.m
        g = m.dist_free[s.pac_pos[:, None], s.ghost_pos].astype(np.float64)
        live = np.where(s.frightened, np.inf, g).min(1)
        small_left = (s.dots & ~m.power0).any(1)
        return (live > self.reserve_range) & small_left

    def _pac_rollout(self, s, noise=None):
        """Cheap greedy Pac-Man: keep away from live ghosts, like dots, go after frightened ghosts."""
        m = self.m
        R = s.B
        if noise is None:
            noise = self.rng.random((R, 4))
        nxt = m.nbr[s.pac_pos]  # R x 4
        legal = nxt >= 0
        straight = legal & (np.arange(4) != REVERSE[s.pac_dir][:, None])
        legal = np.where(straight.any(1, keepdims=True), straight, legal)
        n = np.maximum(nxt, 0)
        gd = m.dist_free[n[:, :, None], s.ghost_pos[:, None, :]].astype(np.float64)  # R x 4 x 4
        live = np.where(s.frightened[:, None, :], np.inf, gd).min(-1)
        prey = np.where(s.frightened[:, None, :], gd, np.inf).min(-1)
        score = (3.0 * np.minimum(live, 6.0) + 2.0 * s.dots[np.arange(R)[:, None], n]
                 - np.where(np.isfinite(prey), 0.5 * prey, 0.0) + noise)
        if self.reserve:
            score = score - 8.0 * (m.power0[n] & self._hold_pellets(s)[:, None])
        return np.where(legal, score, -np.inf).argmax(1)

    # --- decision -----------------------------------------------------------------

    def act(self, s) -> np.ndarray:
        sim, m, N = self.sim, self.m, self.n
        B = s.B
        legal0 = m.nbr[s.pac_pos] >= 0  # B x 4
        idx = np.repeat(np.arange(B), 4 * N)
        first = np.tile(np.repeat(np.arange(4), N), B)
        r = type(s)(**{k: v[idx].copy() for k, v in s.__dict__.items()})
        R = r.B
        value = np.zeros(R)
        dead = np.zeros(R, bool)
        stopped = r.done.copy()
        for k in range(self.depth):
            if k == 0:
                a = first
            else:
                noise = None
                if self.crn:  # the same random numbers for all four first moves of a game
                    noise = np.broadcast_to(self.rng.random((B, 1, N, 4)), (B, 4, N, 4)).reshape(R, 4)
                a = self._pac_rollout(r, noise)
            alive = ~stopped
            _, info = sim.step(r, a, self._ghost_chase(r))
            value += alive * (info["ate"] * 1.0 + info["ghosts_eaten"].sum(1) * self.ghost_bonus
                              + info["cleared"] * 200.0)
            dead |= alive & info["captured"]
            stopped |= r.done
        value -= dead * self.death_cost
        # Break ties between equally safe, equally fed rollouts: end closer to a dot.
        dist = np.where(r.dots, m.dist_free[r.pac_pos].astype(np.float64), np.inf).min(1)
        value -= self.dot_pull * np.where(np.isfinite(dist), dist, 0.0)
        q = value.reshape(B, 4, N).mean(-1)
        q = q + self.heading_bonus * (np.arange(4) == s.pac_dir[:, None])
        if self.reserve:  # do not step onto a power pellet unless forced or a ghost is near
            nxt = np.maximum(m.nbr[s.pac_pos], 0)
            skip_pellet = m.power0[nxt] & self._hold_pellets(s)[:, None]
            other = (legal0 & ~skip_pellet).any(1, keepdims=True)
            legal0 = np.where(other, legal0 & ~skip_pellet, legal0)
        q = np.where(legal0, q, -np.inf) + self.rng.random((B, 4)) * 1e-6
        a = q.argmax(1)
        if self.eps > 0:
            rnd = self.rng.random(B) < self.eps
            pick = np.where(legal0, self.rng.random((B, 4)), -1).argmax(1)
            a = np.where(rnd, pick, a)
        return a


class UCTPlanner(SearchPlanner):
    """UCT tree search over single-tick Pac-Man moves, batched over games.

    One tree per game, all trees grown in lockstep with array operations. Nodes hold the real
    simulator state, so nothing about speeds, skips, pellets or the clock is approximated; chasing
    ghosts are the same neutral path-chasers as in `SearchPlanner`. Leaves are valued by a short
    greedy rollout. Reversing is only allowed at the root (and when it is the only way on).
    """

    def __init__(self, sim, iterations: int = 48, depth: int = 20, c: float = 12.0, seed: int = 0,
                 death_cost: float = 60.0, ghost_bonus: float = 6.0, dot_pull: float = 0.3,
                 eps: float = 0.0, straight_bonus: float = 0.7, dot_weight: float = 0.3):
        super().__init__(sim, n_rollouts=1, depth=depth, seed=seed, death_cost=death_cost,
                         ghost_bonus=ghost_bonus, dot_pull=dot_pull, eps=eps,
                         reserve=False, crn=False, heading_bonus=0.0)
        self.iters, self.c = iterations, c
        self.straight_bonus, self.dot_weight = straight_bonus, dot_weight
        self._pool = None

    # --- smarter rollout moves: straight-line habit and a pull towards the nearest dot ----

    def _pac_rollout(self, s, noise=None):
        m = self.m
        R = s.B
        nxt = m.nbr[s.pac_pos]
        legal = nxt >= 0
        straight = legal & (np.arange(4) != REVERSE[s.pac_dir][:, None])
        legal = np.where(straight.any(1, keepdims=True), straight, legal)
        n = np.maximum(nxt, 0)
        gd = m.dist_free[n[:, :, None], s.ghost_pos[:, None, :]].astype(np.float64)  # R x 4 x 4
        live = np.where(s.frightened[:, None, :], np.inf, gd).min(-1)
        prey = np.where(s.frightened[:, None, :], gd, np.inf).min(-1)
        dd = np.where(s.dots[:, None, :], m.dist_free[n].astype(np.float64), np.inf).min(-1)  # R x 4
        dd = np.where(np.isfinite(dd), dd, 0.0)
        score = (3.0 * np.minimum(live, 7.0) + 2.0 * s.dots[np.arange(R)[:, None], n]
                 - self.dot_weight * np.minimum(dd, 30.0)
                 - np.where(np.isfinite(prey), 0.5 * prey, 0.0)
                 + self.straight_bonus * (np.arange(4) == s.pac_dir[:, None])
                 + self.rng.random((R, 4)))
        return np.where(legal, score, -np.inf).argmax(1)

    # --- tree -----------------------------------------------------------------------

    def _reward(self, info):
        return (info["ate"] * 1.0 + info["ghosts_eaten"].sum(1) * self.ghost_bonus
                + info["cleared"] * 200.0 - info["captured"] * self.death_cost)

    def _legal_children(self, st, prev_dir):
        """B x 4 moves a node may try: legal, and not turning back unless nothing else is open."""
        legal = self.m.nbr[st.pac_pos] >= 0
        fwd = legal & (np.arange(4) != REVERSE[prev_dir][:, None])
        return np.where(fwd.any(1, keepdims=True), fwd, legal)

    def _gather(self, idx):
        return type(self._pool)(**{k: v[idx].copy() for k, v in self._pool.__dict__.items()})

    def _scatter(self, idx, st):
        for k, v in st.__dict__.items():
            self._pool.__dict__[k][idx] = v

    def act(self, s) -> np.ndarray:
        sim, m = self.sim, self.m
        B, C = s.B, self.iters + 1
        if self._pool is None or self._pool_B != B:
            self._pool, self._pool_B = sim.reset(B * C), B
        root = np.arange(B) * C
        # fresh tree: node 0 of every game is the real state
        self._scatter(root, s)
        N = B * C
        parent = np.full(N, -1)
        reward_in = np.zeros(N)
        visits = np.zeros(N)
        value = np.zeros(N)
        terminal = np.zeros(N, bool)
        children = np.full((N, 4), -1)
        lg = np.zeros((N, 4), bool)
        cnt = np.ones(B, dtype=np.int64)
        terminal[root] = s.done
        lg[root] = m.nbr[s.pac_pos] >= 0  # the root may reverse
        ar = np.arange(B)

        for _ in range(self.iters):
            # 1. selection: descend while the node is fully expanded
            cur = root.copy()
            for _d in range(self.iters + 1):
                unexp = (children[cur] == -1) & lg[cur]
                go = ~terminal[cur] & ~unexp.any(1)
                if not go.any():
                    break
                ch = children[cur]
                nc = visits[np.maximum(ch, 0)]
                q = value[np.maximum(ch, 0)] / np.maximum(nc, 1)
                ucb = q + self.c * np.sqrt(np.log(visits[cur] + 1.0)[:, None] / np.maximum(nc, 1))
                a = np.where(ch >= 0, ucb, -np.inf).argmax(1)
                cur = np.where(go, ch[ar, a], cur)
            # 2. expansion
            unexp = (children[cur] == -1) & lg[cur]
            exp = ~terminal[cur] & unexp.any(1)
            g = np.flatnonzero(exp)
            leaf = cur.copy()
            if len(g):
                a = unexp[g].argmax(1)
                new = g * C + cnt[g]
                cnt[g] += 1
                st = self._gather(cur[g])
                prev = st.pac_dir.copy()
                _, info = sim.step(st, a, self._ghost_chase(st))
                self._scatter(new, st)
                parent[new], reward_in[new] = cur[g], self._reward(info)
                terminal[new] = st.done
                lg[new] = self._legal_children(st, st.pac_dir)
                children[cur[g], a] = new
                leaf[g] = new
            # 3. rollout from the leaf
            r = self._gather(leaf)
            ret = np.zeros(B)
            alive = ~r.done
            for _r in range(self.depth):
                if not alive.any():
                    break
                _, info = sim.step(r, self._pac_rollout(r), self._ghost_chase(r))
                ret += alive * self._reward(info)
                alive &= ~r.done
            dist = np.where(r.dots, m.dist_free[r.pac_pos].astype(np.float64), np.inf).min(1)
            ret -= self.dot_pull * np.where(np.isfinite(dist), dist, 0.0) * (~terminal[leaf])
            # 4. backpropagation
            node, act = leaf.copy(), np.ones(B, bool)
            for _b in range(self.iters + 2):
                value[node[act]] += ret[act]
                visits[node[act]] += 1
                ret = ret + np.where(act, reward_in[node], 0.0)
                act = act & (node != root)
                if not act.any():
                    break
                node = np.where(act, parent[node], node)

        # choose the most visited root move (ties: higher mean)
        ch = children[root]
        nv = np.where(ch >= 0, visits[np.maximum(ch, 0)], -1.0)
        mean = np.where(ch >= 0, value[np.maximum(ch, 0)] / np.maximum(nv, 1), -np.inf)
        a = (nv + 1e-6 * mean + self.rng.random((B, 4)) * 1e-9).argmax(1)
        a = np.where((nv.max(1) < 0), 0, a)
        if self.eps > 0:
            rnd = self.rng.random(B) < self.eps
            legal0 = m.nbr[s.pac_pos] >= 0
            pick = np.where(legal0, self.rng.random((B, 4)), -1).argmax(1)
            a = np.where(rnd, pick, a)
        return a
