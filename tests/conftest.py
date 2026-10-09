import numpy as np
import pytest

from ghostrl.maze import load_maze, parse_maze
from ghostrl.sim import Sim, SimConfig

NO_SCATTER = SimConfig(scatter_ticks=(0, 0, 0, 0))


@pytest.fixture(scope="session")
def arcade():
    return load_maze("arcade")


@pytest.fixture(scope="session")
def small():
    return load_maze("small")


def hand_maze(text: str):
    """Parse a hand-drawn maze. Four ghosts are parked in a sealed corridor below
    the drawing (tests then place them by hand); `text` needs exactly one P.
    Rows must be at least 6 wide. Coordinates of the drawing are unchanged."""
    rows = text.rstrip("\n").splitlines()
    W = len(rows[0])
    rows += ["#" * W, "#GGGG" + "#" * (W - 5), "#" * W]
    return parse_maze("\n".join(rows))


def tile(maze, r, c):
    return int(maze.tile_id[r, c])


def place(state, maze, pac=None, pac_dir=None, ghosts=None, ghost_dirs=None, tick=None):
    """Set positions for game 0 by (row, col)."""
    if pac is not None:
        state.pac_pos[0] = tile(maze, *pac)
    if pac_dir is not None:
        state.pac_dir[0] = pac_dir
    for g, rc in (ghosts or {}).items():
        state.ghost_pos[0, g] = tile(maze, *rc)
    for g, d in (ghost_dirs or {}).items():
        state.ghost_dir[0, g] = d
    if tick is not None:
        state.tick[0] = tick


# Ghosts circle a ring; Pac-Man lives in a separate corridor and can never be caught.
RING = """\
########
#GGGG..#
#.####.#
#.####.#
#......#
########
#P.....#
########
"""

# A long corridor for hand-placed capture tests.
CORRIDOR = """\
#############
#P..........#
#############
"""

# Plus-shaped junction for tie-break tests. Arms are one tile long.
PLUS = """\
#######
###.###
##...##
###.###
###P###
#######
"""

# A fork where the straight-line-closer branch is the longer way round.
# A ghost entering from the spur at the top, heading down, must pick left or right.
FORK = """\
####.######
#.........#
#.#######.#
#.#######.#
#....P....#
###########
"""
