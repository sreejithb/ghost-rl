"""One full classic-ghost game on the arcade maze, played by a fixed pseudo-random Pac-Man.

The Pac-Man policy is a hand-rolled integer hash (no RNG library), so the
recorded game depends only on the simulator and the classic rules.
"""

import numpy as np

from ghostrl.maze import REVERSE, load_maze
from ghostrl.sim import Sim

GOLDEN_PATH = "tests/golden/classic_game.npz"  # short scatter-only game (evader below)
FULL_PATH = "tests/golden/classic_full_game.npz"  # long game replayed from a recorded action log
SEED = 0  # picked so the game outlasts the whole scatter/chase schedule (see make_golden.py)


def _mix(x: int) -> int:
    x = (x * 2654435761 + 0x9E3779B9) % 2**32
    x ^= x >> 15
    x = (x * 2246822519) % 2**32
    x ^= x >> 13
    return x


def pac_action(sim, state, tick: int, seed: int = SEED) -> int:
    """Evader: step to the tile farthest (by path) from the nearest ghost, preferring
    tiles with dots, never reversing unless forced. Ties are broken by the hash."""
    m = sim.maze
    legal = sim.pac_legal(state)[0].copy()
    straight = legal & (np.arange(4) != REVERSE[state.pac_dir[0]])
    options = np.flatnonzero(straight if straight.any() else legal)
    nxt = m.nbr[state.pac_pos[0], options]
    gap = m.dist_free[nxt][:, state.ghost_pos[0]].min(1).astype(int)
    score = 4 * gap + state.dots[0, nxt]
    best = options[score == score.max()]
    return int(best[_mix(tick + 100_003 * seed) % len(best)])


def play_golden(seed: int = SEED) -> dict[str, np.ndarray]:
    sim = Sim(load_maze("arcade"))
    s = sim.reset(1)
    pac, ghosts, rewards = [], [], []
    while not s.done[0]:
        r, info = sim.step(s, np.array([pac_action(sim, s, int(s.tick[0]), seed)]))
        pac.append(s.pac_pos[0])
        ghosts.append(s.ghost_pos[0].copy())
        rewards.append(r[0])
    outcome = int(info["captured"][0]) + 2 * int(info["cleared"][0]) + 4 * int(info["timeout"][0])
    return dict(pac_pos=np.array(pac), ghost_pos=np.array(ghosts), rewards=np.array(rewards),
                dots_left=np.array(int(s.dots.sum())), outcome=np.array(outcome))


def play_actions(pac_actions) -> dict[str, np.ndarray]:
    """Replay a recorded Pac-Man action log (classic ghosts). No planner, no policy: only the sim runs."""
    sim = Sim(load_maze("arcade"))
    s = sim.reset(1)
    pac, ghosts, rewards, power, eaten, switched = [], [], [], [], [], []
    for a in pac_actions:
        if s.done[0]:
            raise AssertionError("action log is longer than the game")
        r, info = sim.step(s, np.array([a]))
        pac.append(s.pac_pos[0])
        ghosts.append(s.ghost_pos[0].copy())
        rewards.append(r[0])
        power.append(info["power"][0])
        eaten.append(info["ghosts_eaten"][0].copy())
        switched.append(info["switched"][0])
    outcome = int(info["captured"][0]) + 2 * int(info["cleared"][0]) + 4 * int(info["timeout"][0])
    return dict(pac_actions=np.asarray(pac_actions, dtype=np.int8), pac_pos=np.array(pac),
                ghost_pos=np.array(ghosts), rewards=np.array(rewards), power=np.array(power),
                eaten=np.array(eaten), switched=np.array(switched),
                dots_left=np.array(int(s.dots.sum())), outcome=np.array(outcome))
