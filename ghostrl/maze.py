"""ASCII maze parsing and the tables derived from it.

Conventions: row 0 is the top, "up" is row - 1. Directions are always
up, left, down, right (indices 0-3); the ghost tie-break depends on it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

UP, LEFT, DOWN, RIGHT = 0, 1, 2, 3
DIR_VEC = np.array([(-1, 0), (0, -1), (1, 0), (0, 1)], dtype=np.int64)  # (row, col)
REVERSE = np.array([DOWN, RIGHT, UP, LEFT], dtype=np.int64)

GHOST_NAMES = ("blinky", "pinky", "inky", "clyde")
UNREACHABLE = np.int16(32767)

MAZE_DIR = Path(__file__).resolve().parent.parent / "mazes"


@dataclass(eq=False)
class Maze:
    """A parsed maze. T is the number of walkable tiles."""

    grid: np.ndarray  # H x W array of single characters
    corners: np.ndarray  # 4 x 2, (row, col) scatter targets, may lie outside the maze
    name: str = ""

    H: int = field(init=False)
    W: int = field(init=False)
    T: int = field(init=False)
    tile_id: np.ndarray = field(init=False)  # H x W, tile index or -1
    xy: np.ndarray = field(init=False)  # T x 2, (row, col)
    nbr: np.ndarray = field(init=False)  # T x 4, neighbour tile or -1
    dots0: np.ndarray = field(init=False)  # T bool, dots at the start of a game
    power0: np.ndarray = field(init=False)  # T bool, energizer tiles (also dots)
    pac_start: int = field(init=False)
    ghost_start: np.ndarray = field(init=False)  # 4 tile indices, Blinky..Clyde

    def __post_init__(self) -> None:
        g = self.grid
        self.H, self.W = g.shape
        walkable = g != "#"
        self.tile_id = np.full(g.shape, -1, dtype=np.int64)
        self.xy = np.argwhere(walkable).astype(np.int64)  # row-major order
        self.T = len(self.xy)
        self.tile_id[walkable] = np.arange(self.T)

        self.nbr = np.full((self.T, 4), -1, dtype=np.int64)
        for t, (r, c) in enumerate(self.xy):
            for d, (dr, dc) in enumerate(DIR_VEC):
                nr, nc = r + dr, c + dc
                if nc < 0 or nc >= self.W:  # horizontal tunnel wrap
                    nc %= self.W
                if 0 <= nr < self.H:
                    self.nbr[t, d] = self.tile_id[nr, nc]

        tiles = g[walkable]
        self.dots0 = (tiles == ".") | (tiles == "o")
        self.power0 = tiles == "o"  # a plain dot unless SimConfig.power_pellets is on
        starts = np.argwhere(g == "P")
        if len(starts) != 1:
            raise ValueError(f"maze needs exactly one P, found {len(starts)}")
        self.pac_start = int(self.tile_id[tuple(starts[0])])
        gs = np.argwhere(g == "G")  # row-major reading order = Blinky, Pinky, Inky, Clyde
        if len(gs) != 4:
            raise ValueError(f"maze needs exactly four G, found {len(gs)}")
        self.ghost_start = np.array([self.tile_id[tuple(p)] for p in gs], dtype=np.int64)

    @cached_property
    def dist_free(self) -> np.ndarray:
        """T x T shortest path in ticks, reversing allowed (Pac-Man)."""
        out = np.full((self.T, self.T), UNREACHABLE, dtype=np.int16)
        for src in range(self.T):
            row = out[src]
            row[src] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self.nbr[u]:
                    if v >= 0 and row[v] == UNREACHABLE:
                        row[v] = row[u] + 1
                        q.append(v)
        return out

    @cached_property
    def dist_ghost(self) -> np.ndarray:
        """(T*4) x T shortest path from (tile, heading) to a tile, no reversing.

        Row index is tile * 4 + heading. The first step may not reverse the
        heading (unless it is the only way out); the same holds at every
        later tile. The planner uses this to predict ghosts.
        """
        T = self.T
        # succ[s, d]: state reached from state s = tile*4 + heading by moving d.
        succ = np.full((T * 4, 4), -1, dtype=np.int64)
        for t in range(T):
            for h in range(4):
                moves = [d for d in range(4) if self.nbr[t, d] >= 0 and d != REVERSE[h]]
                if not moves:
                    moves = [d for d in range(4) if self.nbr[t, d] >= 0]
                for d in moves:
                    succ[t * 4 + h, d] = self.nbr[t, d] * 4 + d
        out = np.full((T * 4, T), UNREACHABLE, dtype=np.int16)
        for src in range(T * 4):
            seen = np.zeros(T * 4, dtype=bool)
            out[src, src // 4] = 0
            seen[src] = True
            q = deque([(src, 0)])
            while q:
                s, k = q.popleft()
                for n in succ[s]:
                    if n >= 0 and not seen[n]:
                        seen[n] = True
                        if out[src, n // 4] == UNREACHABLE:
                            out[src, n // 4] = k + 1
                        q.append((n, k + 1))
        return out


def parse_maze(text: str, name: str = "", default_corners=None) -> Maze:
    """Parse maze text. Lines starting with ';' are header/comments.

    ';scatter <ghost> <row> <col>' sets a scatter corner.
    """
    corners = {}
    rows = []
    for line in text.splitlines():
        if line.startswith(";scatter"):
            _, ghost, r, c = line.split()
            corners[ghost.lower()] = (int(r), int(c))
        elif line.startswith(";"):
            continue
        elif line.strip("\r") != "" or rows:
            rows.append(line.rstrip("\r"))
    while rows and rows[-1] == "":
        rows.pop()
    widths = {len(r) for r in rows}
    if len(widths) != 1:
        raise ValueError(f"ragged maze rows: widths {sorted(widths)}")
    grid = np.array([list(r) for r in rows])
    if corners:
        missing = [n for n in GHOST_NAMES if n not in corners]
        if missing:
            raise ValueError(f"missing scatter corners for {missing}")
        arr = np.array([corners[n] for n in GHOST_NAMES], dtype=np.int64)
    elif default_corners is not None:
        arr = np.asarray(default_corners, dtype=np.int64)
    else:
        arr = np.array([(0, grid.shape[1]), (0, 0), (grid.shape[0], grid.shape[1]), (grid.shape[0], 0)])
    return Maze(grid=grid, corners=arr, name=name)


def load_maze(name: str) -> Maze:
    """Load mazes/<name>.txt."""
    return parse_maze((MAZE_DIR / f"{name}.txt").read_text(), name=name)
