"""RL environment around `Sim`: one side is learned, the other is a fixed controller.

`controls="pacman"`: actions are B ints, the ghosts follow `opponent` (default: classic rules).
`controls="ghosts"`: actions are B x 4 ints (chase mode only; scatter and frightened stay scripted),
Pac-Man follows `opponent`, a callable `state -> B actions`.

Both sides share `Sim.step`; this wrapper only decides who supplies which actions, what is
observed and how it is rewarded. Finished games reset automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .maze import DIR_VEC
from .sim import Sim, State

N_PLANES = 10  # see PLANES
PLANES = ("walls", "dots", "power", "pacman", "ghosts", "frightened",
          "ghost_up", "ghost_left", "ghost_down", "ghost_right")
N_SCALARS = 9  # pac heading (4), fright left, scatter, ticks to switch, dots left, time used
# history=True adds 4 trail planes (ghosts 1 and 2 ticks ago, Pac-Man 1 and 2 ticks ago) and 2 scalars
# (position in the 16-tick skip cycle, share of ghosts with a pending reversal).
HIST_PLANES = ("ghosts_1_ago", "ghosts_2_ago", "pacman_1_ago", "pacman_2_ago")
HIST_SCALARS = 2


def obs_dims(history: bool) -> tuple[int, int]:
    return N_PLANES + 4 * history, N_SCALARS + 2 * history


@dataclass(frozen=True)
class PacmanReward:
    dot: float = 0.1
    power: float = 0.5  # on top of the dot
    ghost_eaten: float = 1.0
    death: float = -5.0
    clear: float = 10.0
    tick: float = -0.005


def pacman_reward(info: dict, r: PacmanReward) -> np.ndarray:
    return (r.dot * info["ate"] + r.power * info["power"] + r.ghost_eaten * info["ghosts_eaten"].sum(1)
            + r.death * info["captured"] + r.clear * info["cleared"] + r.tick * ~info["was_done"]) * ~info["was_done"]


class GhostRLEnv:
    def __init__(self, sim: Sim, B: int, controls: str = "pacman", opponent=None,
                 pac_reward: PacmanReward | None = None, history: bool = False):
        if controls not in ("pacman", "ghosts"):
            raise ValueError("controls must be 'pacman' or 'ghosts'")
        self.sim, self.B, self.controls, self.opponent = sim, B, controls, opponent
        self.pac_reward = pac_reward or PacmanReward()
        self.history = history
        self.n_planes, self.n_scalars = obs_dims(history)
        self._prev = np.full((2, B, 5), -1, dtype=np.int64)  # tile 1 and 2 ticks ago: pac, ghost x4 (-1 = none)
        m = sim.maze
        self.walls = (m.grid == "#").astype(np.float32)
        self._flat = (m.xy[:, 0] * m.W + m.xy[:, 1]).astype(np.int64)  # tile -> flat cell
        self.state: State = sim.reset(B)
        self.ep_return = np.zeros(B)
        self.ep_len = np.zeros(B, dtype=np.int64)

    # --- observation ---------------------------------------------------------------

    def observe(self) -> dict:
        """planes B x 10 x H x W, scalars B x 9, mask (B x 4 for Pac-Man, B x 4 x 4 for ghosts)."""
        s, m, c = self.state, self.sim.maze, self.sim.cfg
        B, H, W = self.B, m.H, m.W
        planes = np.zeros((B, self.n_planes, H * W), dtype=np.float32)
        ar = np.arange(B)
        planes[:, 0] = self.walls.reshape(-1)
        dots = s.dots & ~m.power0
        planes[:, 1, self._flat] = dots
        planes[:, 2, self._flat] = s.dots & m.power0
        planes[ar, 3, self._flat[s.pac_pos]] = 1
        gcell = self._flat[s.ghost_pos]  # B x 4
        live = ~s.frightened
        for g in range(4):
            planes[ar, 4, gcell[:, g]] += live[:, g]
            planes[ar, 5, gcell[:, g]] += s.frightened[:, g]
            planes[ar, 6 + s.ghost_dir[:, g], gcell[:, g]] += live[:, g]
        mt = self.sim.mode_time(s)
        scalars = np.zeros((B, self.n_scalars), dtype=np.float32)
        scalars[ar, s.pac_dir] = 1
        scalars[:, 4] = s.fright_left / max(c.fright_ticks, 1)
        scalars[:, 5] = self.sim.is_scatter(mt)
        scalars[:, 6] = np.minimum(self.sim.ticks_to_switch(mt), 300) / 300
        scalars[:, 7] = s.dots.sum(1) / max(m.dots0.sum(), 1)
        scalars[:, 8] = s.tick / c.max_ticks
        if self.history:
            for k in range(2):  # k = 0: one tick ago, k = 1: two ticks ago
                prev = self._prev[k]
                for g in range(4):
                    ok = prev[:, 1 + g] >= 0
                    planes[ar[ok], N_PLANES + k, self._flat[prev[ok, 1 + g]]] += 1
                ok = prev[:, 0] >= 0
                planes[ar[ok], N_PLANES + 2 + k, self._flat[prev[ok, 0]]] = 1
            scalars[:, N_SCALARS] = (s.tick % 16) / 15
            scalars[:, N_SCALARS + 1] = s.ghost_rev.mean(1)
        out = dict(planes=planes.reshape(B, self.n_planes, H, W), scalars=scalars)
        if self.controls == "pacman":
            out["mask"] = self.sim.pac_legal(s)
        else:
            out["mask"] = self.sim.ghost_legal(s)
            out["ghost_id"] = np.broadcast_to(np.eye(4, dtype=np.float32), (B, 4, 4))
        return out

    # --- stepping ------------------------------------------------------------------

    def reset(self) -> dict:
        self.state = self.sim.reset(self.B)
        self.ep_return[:] = 0
        self.ep_len[:] = 0
        self._prev[:] = -1
        return self.observe()

    def step(self, actions):
        """Returns obs, reward, done, info. Reward is B for Pac-Man, B for the shared ghost team.

        `done` marks games that finished on this step; their obs is already the fresh game's.
        info["final"] has the episode outcome for those games.
        """
        sim, s = self.sim, self.state
        actions = np.asarray(actions, dtype=np.int64)
        was_done = s.done.copy()
        if self.controls == "pacman":
            pac, ghosts = actions, (None if self.opponent is None else self.opponent(s))
        else:
            pac, ghosts = (self.opponent(s) if self.opponent is not None else sim_zero(self.B)), actions
        before = np.concatenate([s.pac_pos[:, None], s.ghost_pos], 1)
        sim_reward, info = sim.step(s, pac, ghosts)
        self._prev[1], self._prev[0] = self._prev[0], before
        self._prev[:, was_done] = -1
        info["was_done"] = was_done
        reward = sim_reward if self.controls == "ghosts" else pacman_reward(info, self.pac_reward)
        self.ep_return += reward
        self.ep_len += ~was_done
        done = s.done & ~was_done
        final = dict(done=done, ret=self.ep_return.copy(), length=self.ep_len.copy(),
                     captured=info["captured"] & done, cleared=info["cleared"] & done,
                     timeout=info["timeout"] & done, dots_left=s.dots.sum(1))
        if done.any():
            sim.reset_where(s, s.done)
            self._prev[:, done] = -1
            self.ep_return[done] = 0
            self.ep_len[done] = 0
        info["final"] = final
        return self.observe(), reward, done, info


def sim_zero(B):
    return np.zeros(B, dtype=np.int64)
