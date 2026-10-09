"""Regenerate the golden snapshots. Run by hand, only when a rule change is intended
and its failing test is already written:  python tests/make_golden.py

The long game is recorded once from the Pac-Man planner and stored as an action log;
the tests replay that log and never call the planner, so planner changes cannot touch it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

from golden_game import FULL_PATH, GOLDEN_PATH, play_actions, play_golden
from ghostrl.maze import load_maze
from ghostrl.pacman import Planner
from ghostrl.sim import Sim

g = play_golden()
Path(GOLDEN_PATH).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(GOLDEN_PATH, **g)
print(f"saved {GOLDEN_PATH}: {len(g['pac_pos'])} ticks, outcome {int(g['outcome'])}")

# Long game: play 40 planner games, keep the first that covers every rule we want to pin down.
sim = Sim(load_maze("arcade"))
B = 40
s = sim.reset(B)
planner = Planner(sim.maze, pellet_bonus=20.0, pellet_bonus_danger=60.0, seed=1)
log, power, eaten = [], np.zeros(B, int), np.zeros(B, int)
while not s.done.all():
    a = planner.act(s)
    log.append(a.copy())
    _, info = sim.step(s, a)
    power += info["power"]
    eaten += info["ghosts_eaten"].sum(1)
log = np.array(log)
ok = np.flatnonzero((s.tick >= 700) & (s.tick < 1800) & (power >= 2) & (eaten >= 2) & (s.dots.sum(1) > 0))
assert len(ok), "no planner game covers the schedule, power pellets and eaten ghosts"
i = int(ok[0])
full = play_actions(log[: s.tick[i], i])
np.savez_compressed(FULL_PATH, **full)
print(f"saved {FULL_PATH}: {len(full['pac_pos'])} ticks, outcome {int(full['outcome'])}, "
      f"{int(full['power'].sum())} pellets, {int(full['eaten'].sum())} ghosts eaten")
