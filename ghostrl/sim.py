"""Batched tile-grid Pac-Man simulator. See docs/simulator-spec.md.

State is arrays with a leading batch dimension B. `Sim.step` mutates the
state in place. Both ghost teams go through the same step(); only the
chase-mode move choice differs (`ghost_action`, default: classic rules).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from . import classic
from .maze import REVERSE, Maze

LEFT = 1


@dataclass(frozen=True)
class SimConfig:
    max_ticks: int = 2000
    # Mode clock, arcade level 1. Scatter lengths S1..S4, chase lengths C1..C3;
    # the fourth chase lasts until the end. Zero-length scatters give "no scatter".
    scatter_ticks: tuple[int, ...] = (53, 53, 38, 38)
    chase_ticks: tuple[int, ...] = (152, 152, 152)
    # Each ghost skips 1 tick in `period` (0 = never). 16 -> 15/16 speed.
    ghost_skip_period: tuple[int, int, int, int] = (16, 16, 16, 16)
    pinky_offset: int = 4
    inky_pivot: int = 2
    clyde_radius_sq: int = 64
    reward_capture: float = 1.0
    reward_clear: float = -1.0
    reward_tick: float = -0.01
    # Frightened mode: energizers frighten every ghost. Scripted for both teams.
    # 45 ticks = 6 s at 7.58 tiles/s (arcade level 1). Frightened ghosts run at 50% speed.
    power_pellets: bool = True
    fright_ticks: int = 45
    fright_skip_period: int = 2
    reward_ghost_eaten: float = -0.5


@dataclass
class State:
    pac_pos: np.ndarray  # B
    pac_dir: np.ndarray  # B
    ghost_pos: np.ndarray  # B x 4
    ghost_dir: np.ndarray  # B x 4
    ghost_rev: np.ndarray  # B x 4 bool, a mode-switch reversal waiting to be executed
    dots: np.ndarray  # B x T bool
    frightened: np.ndarray  # B x 4 bool
    fright_left: np.ndarray  # B, ticks of frightened mode remaining
    paused: np.ndarray  # B, ticks the scatter/chase clock has been frozen
    tick: np.ndarray  # B
    done: np.ndarray  # B bool

    def copy(self) -> "State":
        return State(**{k: v.copy() for k, v in self.__dict__.items()})

    @property
    def B(self) -> int:
        return len(self.tick)

    def hash(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self.__dict__):
            h.update(np.ascontiguousarray(self.__dict__[k]).tobytes())
        return h.hexdigest()


class Sim:
    def __init__(self, maze: Maze, cfg: SimConfig | None = None):
        self.maze = maze
        self.cfg = cfg or SimConfig()
        c = self.cfg
        lengths = [x for pair in zip(c.scatter_ticks, c.chase_ticks) for x in pair]
        lengths.append(c.scatter_ticks[-1])  # S1 C1 S2 C2 S3 C3 S4, then chase forever
        self._phase_start = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        self._skip = np.asarray(c.ghost_skip_period, dtype=np.int64)

    # --- mode clock ---------------------------------------------------------

    def is_scatter(self, tick):
        """Phase 0, 2, 4, 6 are scatter; phase 7 (final chase) never ends."""
        phase = np.searchsorted(self._phase_start, tick, side="right") - 1
        return phase % 2 == 0

    def switch_ticks(self) -> np.ndarray:
        """Ticks at which the mode changes (heading reversal), in order."""
        t = self._phase_start[1:]
        t = t[(t > 0)]
        keep = self.is_scatter(t) != self.is_scatter(t - 1)
        return t[keep]

    def ticks_to_switch(self, tick):
        """Ticks until the next mode switch (observation hook); large when none left."""
        sw = self.switch_ticks()
        i = np.searchsorted(sw, tick, side="right")
        nxt = np.where(i < len(sw), sw[np.minimum(i, len(sw) - 1)], np.iinfo(np.int64).max // 2)
        return nxt - tick

    def mode_time(self, state: State) -> np.ndarray:
        """Tick count on the scatter/chase clock, which stands still while ghosts are frightened."""
        return state.tick - state.paused

    # --- state ---------------------------------------------------------------

    def reset(self, B: int) -> State:
        m = self.maze
        return State(
            pac_pos=np.full(B, m.pac_start, dtype=np.int64),
            pac_dir=np.full(B, LEFT, dtype=np.int64),
            ghost_pos=np.tile(m.ghost_start, (B, 1)),
            ghost_dir=np.full((B, 4), LEFT, dtype=np.int64),
            ghost_rev=np.zeros((B, 4), dtype=bool),
            dots=np.tile(m.dots0, (B, 1)),
            frightened=np.zeros((B, 4), dtype=bool),
            fright_left=np.zeros(B, dtype=np.int64),
            paused=np.zeros(B, dtype=np.int64),
            tick=np.zeros(B, dtype=np.int64),
            done=np.zeros(B, dtype=bool),
        )

    def reset_where(self, state: State, mask: np.ndarray) -> None:
        """Reset the games selected by `mask` in place (autoreset helper)."""
        fresh = self.reset(state.B)
        for k, v in state.__dict__.items():
            v[mask] = fresh.__dict__[k][mask]

    # --- queries used by controllers ----------------------------------------

    def pac_legal(self, state: State) -> np.ndarray:
        return self.maze.nbr[state.pac_pos] >= 0  # B x 4

    def ghost_legal(self, state: State) -> np.ndarray:
        """B x 4 x 4 mask of moves a ghost may choose (walls and reversing masked)."""
        return classic.legal_moves(self.maze.nbr, state.ghost_pos, state.ghost_dir)

    def classic_targets(self, state: State, scatter: np.ndarray) -> np.ndarray:
        m, c = self.maze, self.cfg
        pac_rc = m.xy[state.pac_pos]
        chase = classic.chase_targets(
            pac_rc, state.pac_dir, m.xy[state.ghost_pos], m.corners,
            c.pinky_offset, c.inky_pivot, c.clyde_radius_sq)
        corners = classic.scatter_targets(m.corners, (state.B,))
        return np.where(scatter[:, None, None], corners, chase)

    def classic_actions(self, state: State, scatter: np.ndarray | None = None) -> np.ndarray:
        """B x 4 moves the classic rules would choose right now."""
        if scatter is None:
            scatter = self.is_scatter(self.mode_time(state))
        tgt = self.classic_targets(state, scatter)
        return classic.greedy_move(self.maze.nbr, self.maze.xy, state.ghost_pos, state.ghost_dir, tgt)

    # --- one tick -------------------------------------------------------------

    def step(self, state: State, pac_action: np.ndarray, ghost_action: np.ndarray | None = None):
        """Advance every unfinished game by one tick, in place.

        pac_action: B ints. An illegal move is ignored (Pac-Man stays put).
        ghost_action: B x 4 ints, used in chase mode only; scatter is scripted
        and a pending mode-switch reversal always overrides. None = classic.
        Returns (reward B, info dict).
        """
        m, c = self.maze, self.cfg
        B = state.B
        active = ~state.done
        t = state.tick
        ar = np.arange(B)

        # 1. Mode clock: it freezes while frightened. On a switch every ghost's heading reverses.
        paused_old = state.paused
        state.paused = paused_old + (active & (state.fright_left > 0))
        mt = t - state.paused
        scatter = self.is_scatter(mt)
        switched = active & (t > 0) & (scatter != self.is_scatter(np.maximum(t - 1 - paused_old, 0)))
        state.ghost_dir = np.where(switched[:, None], REVERSE[state.ghost_dir], state.ghost_dir)
        state.ghost_rev |= switched[:, None]

        # 2. Snapshot: both controllers read the state as it is now.
        pac_action = np.asarray(pac_action, dtype=np.int64)
        classic_move = self.classic_actions(state, scatter)
        if ghost_action is None:
            chosen = classic_move
        else:
            chosen = np.where(scatter[:, None], classic_move, np.asarray(ghost_action, dtype=np.int64))
        # Frightened ghosts wander: scripted for both teams, whatever the controller says.
        chosen = np.where(state.frightened, self._fright_move(state), chosen)
        # A pending reversal means "keep going in the (already flipped) heading".
        gnxt = m.nbr[state.ghost_pos]  # B x 4 x 4
        rev_ok = state.ghost_rev & (np.take_along_axis(gnxt, state.ghost_dir[..., None], -1)[..., 0] >= 0)
        chosen = np.where(rev_ok, state.ghost_dir, chosen)

        # Ghost speed: skip tick when t % p == p - 1 (identical for both teams).
        p = np.where(state.frightened, c.fright_skip_period, self._skip)
        skip = (p > 0) & ((t[:, None] % np.maximum(p, 1)) == p - 1)
        g_moving = active[:, None] & ~skip

        legal = self.ghost_legal(state)
        bad = g_moving & ~rev_ok & ~np.take_along_axis(legal, chosen[..., None], -1)[..., 0]
        if bad.any():
            raise ValueError("illegal ghost action (wall or reverse)")

        # 3. Move. Pac-Man first for bookkeeping, ghosts from the same snapshot.
        pac_old, ghost_old = state.pac_pos.copy(), state.ghost_pos.copy()
        pac_next = m.nbr[state.pac_pos, pac_action]
        pac_moving = active & (pac_next >= 0)
        state.pac_pos = np.where(pac_moving, pac_next, state.pac_pos)
        state.pac_dir = np.where(pac_moving, pac_action, state.pac_dir)
        ate = pac_moving & state.dots[ar, state.pac_pos]
        power = ate & c.power_pellets & m.power0[state.pac_pos]
        state.dots[ar, state.pac_pos] &= ~pac_moving

        g_next = np.take_along_axis(gnxt, chosen[..., None], -1)[..., 0]
        state.ghost_pos = np.where(g_moving, g_next, state.ghost_pos)
        state.ghost_dir = np.where(g_moving, chosen, state.ghost_dir)
        forced = g_moving & rev_ok
        state.ghost_rev &= ~g_moving

        # 4. Power pellet: every ghost turns frightened and about-faces, before collisions are judged.
        state.frightened |= power[:, None]
        state.fright_left = np.where(power, c.fright_ticks, state.fright_left)
        state.ghost_dir = np.where(power[:, None], REVERSE[state.ghost_dir], state.ghost_dir)
        state.ghost_rev |= power[:, None]

        # 5. Contact: a frightened ghost is eaten and returns to its start; any other ghost catches Pac-Man.
        hit = (state.ghost_pos == state.pac_pos[:, None]) | (
            g_moving & pac_moving[:, None]
            & (state.ghost_pos == pac_old[:, None]) & (ghost_old == state.pac_pos[:, None]))
        eaten = active[:, None] & hit & state.frightened
        captured = active & (hit & ~state.frightened).any(1)
        state.ghost_pos = np.where(eaten, m.ghost_start, state.ghost_pos)
        state.ghost_dir = np.where(eaten, LEFT, state.ghost_dir)
        state.ghost_rev &= ~eaten
        state.frightened &= ~eaten

        # 6. End: capture (wins over clearing on the same tick), last dot, tick limit.
        cleared = active & ate & ~state.dots.any(1) & ~captured
        state.tick = state.tick + active
        timeout = active & (state.tick >= c.max_ticks) & ~captured & ~cleared
        state.done |= captured | cleared | timeout

        # Frightened mode counts down; the tick the pellet was eaten is not charged.
        counting = active & ~power & (state.fright_left > 0)
        state.fright_left = state.fright_left - counting
        state.frightened &= (state.fright_left > 0)[:, None]

        reward = active * (c.reward_tick + c.reward_capture * captured + c.reward_clear * cleared
                           + c.reward_ghost_eaten * eaten.sum(1))
        info = dict(captured=captured, cleared=cleared, timeout=timeout, switched=switched,
                    scatter=scatter, ate=ate, power=power, ghosts_eaten=eaten, pac_moved=pac_moving,
                    ghost_moved=g_moving, forced_reverse=forced, pac_old=pac_old, ghost_old=ghost_old)
        return reward, info

    def _fright_move(self, state: State) -> np.ndarray:
        """B x 4 scripted frightened moves: a hash-picked legal move (the arcade uses a pseudo-random one).

        Depends only on (tick, ghost, tile, heading), so batching and replays stay exact.
        """
        legal = self.ghost_legal(state)
        h = (state.tick[:, None].astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
             + np.arange(4, dtype=np.uint64) * np.uint64(0xBF58476D1CE4E5B9)
             + state.ghost_pos.astype(np.uint64) * np.uint64(0x94D049BB133111EB)
             + state.ghost_dir.astype(np.uint64))
        h ^= h >> np.uint64(31)
        h *= np.uint64(0xD6E8FEB86659FD93)
        h ^= h >> np.uint64(29)
        k = (h % np.maximum(legal.sum(-1), 1).astype(np.uint64)).astype(np.int64)
        pick = legal & (np.cumsum(legal, -1) - 1 == k[..., None])
        return pick.argmax(-1)
