"""Hyperparameter search for the flat Monte-Carlo Pac-Man planner (`SearchPlanner`).

A config is a dict of SearchPlanner arguments. `run_config` plays a batch of games against the chosen
ghosts and returns the same numbers as the results table. `sample_configs` draws random configs.
"""

from __future__ import annotations

import time

import numpy as np

from .env import PacmanReward, pacman_reward
from .maze import load_maze
from .pacman import SearchPlanner
from .sim import Sim

BASELINE = dict(n_rollouts=8, depth=25, death_cost=60.0, ghost_bonus=6.0, dot_pull=0.3, fear_w=3.0, fear_cap=6.0,
                dotnext_w=2.0, noise=1.0, heading_bonus=0.3, reserve=False)
SPACE = dict(
    n_rollouts=[4, 8, 16], depth=[15, 25, 40], death_cost=[30.0, 60.0, 120.0, 240.0], ghost_bonus=[0.0, 3.0, 6.0, 12.0],
    dot_pull=[0.0, 0.05, 0.15, 0.3, 0.6], fear_w=[1.5, 3.0, 6.0], fear_cap=[4.0, 6.0, 10.0], dotnext_w=[1.0, 2.0, 4.0],
    noise=[0.5, 1.0, 2.0], heading_bonus=[0.0, 0.3, 1.0], reserve=[False, True])


def sample_configs(n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    out, seen = [], {tuple(sorted(BASELINE.items()))}
    while len(out) < n:
        c = {k: v[rng.integers(len(v))] for k, v in SPACE.items()}
        c = {k: (v.item() if hasattr(v, "item") else v) for k, v in c.items()}
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def ghost_policy(kind: str, sim: Sim, seed: int = 1):
    """None = the classic rules; otherwise a function state -> B x 4 chase actions."""
    if kind == "classic":
        return None
    if kind == "chaser":
        helper = SearchPlanner(sim)
        return helper._ghost_chase
    rng = np.random.default_rng(seed)

    def random_ghosts(s):
        legal = sim.ghost_legal(s)
        return (legal * rng.random(legal.shape)).argmax(-1)
    return random_ghosts


def run_config(cfg: dict, games: int = 400, seed: int = 1, ghosts: str = "classic", maze: str = "arcade") -> dict:
    sim = Sim(load_maze(maze))
    s = sim.reset(games)
    planner = SearchPlanner(sim, seed=seed, eps=0.02, **cfg)
    gp = ghost_policy(ghosts, sim, seed)
    rew = PacmanReward()
    ret, eaten, t0, steps = np.zeros(games), np.zeros(games), time.time(), 0
    while not s.done.all():
        was = s.done.copy()
        _, info = sim.step(s, planner.act(s), None if gp is None else gp(s))
        info["was_done"] = was
        ret += pacman_reward(info, rew)
        eaten += info["ghosts_eaten"].sum(1)
        steps += int((~was).sum())
    dots = s.dots.sum(1)
    timeouts = int(((dots > 0) & (s.tick >= sim.cfg.max_ticks)).sum())
    return dict(cfg=cfg, games=games, ghosts=ghosts, seed=seed, clears=int((dots == 0).sum()),
                dots_left=float(dots.mean()), ticks=float(s.tick.mean()), ret=float(ret.mean()),
                ret_se=float(ret.std() / np.sqrt(games)), ghosts_eaten=float(eaten.mean()),
                timeouts=timeouts, seconds=time.time() - t0, steps_per_s=steps / (time.time() - t0))
