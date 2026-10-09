"""The 1980 ghost rules: target functions and the greedy movement rule.

All functions are pure and broadcast over leading batch dimensions.
Coordinates are (row, col) in the last axis; directions are indices 0-3
(up, left, down, right).
"""

from __future__ import annotations

import numpy as np

from .maze import DIR_VEC, REVERSE, UP

BIG = np.int64(1) << 40


def pinky_target(pac_rc, pac_dir, offset: int = 4):
    """`offset` tiles ahead of Pac-Man; facing up also shifts `offset` left (arcade bug)."""
    pac_rc, pac_dir = np.asarray(pac_rc), np.asarray(pac_dir)
    t = pac_rc + offset * DIR_VEC[pac_dir]
    return t - np.where((pac_dir == UP)[..., None], (0, offset), (0, 0))


def inky_target(pac_rc, pac_dir, blinky_rc, pivot: int = 2):
    """Reflect Blinky through the pivot tile `pivot` ahead of Pac-Man (same up bug)."""
    p = pinky_target(pac_rc, pac_dir, offset=pivot)
    return 2 * p - np.asarray(blinky_rc)


def clyde_target(pac_rc, clyde_rc, corner, radius_sq: int = 64):
    """Chase when at least 8 tiles away (squared distance >= 64), else go home."""
    pac_rc, clyde_rc = np.asarray(pac_rc), np.asarray(clyde_rc)
    d2 = ((pac_rc - clyde_rc) ** 2).sum(-1)
    return np.where((d2 >= radius_sq)[..., None], pac_rc, np.asarray(corner))


def chase_targets(pac_rc, pac_dir, ghost_rc, corners, pinky_offset: int = 4,
                  inky_pivot: int = 2, clyde_radius_sq: int = 64):
    """Chase targets for all four ghosts: (..., 4, 2).

    pac_rc (..., 2), pac_dir (...), ghost_rc (..., 4, 2), corners (4, 2).
    """
    pac_rc, pac_dir, ghost_rc = np.asarray(pac_rc), np.asarray(pac_dir), np.asarray(ghost_rc)
    return np.stack([
        pac_rc,
        pinky_target(pac_rc, pac_dir, pinky_offset),
        inky_target(pac_rc, pac_dir, ghost_rc[..., 0, :], inky_pivot),
        clyde_target(pac_rc, ghost_rc[..., 3, :], corners[3], clyde_radius_sq),
    ], axis=-2)


def scatter_targets(corners, batch_shape=()):
    """Each ghost's corner, broadcast to (*batch_shape, 4, 2)."""
    return np.broadcast_to(np.asarray(corners), (*batch_shape, 4, 2))


def legal_moves(nbr, pos, heading):
    """Legal-move mask (..., 4): non-wall, not the reverse of `heading`.

    If nothing is left (dead end, debug mazes only) reversing is allowed.
    """
    open_ = nbr[pos] >= 0
    legal = open_ & (np.arange(4) != REVERSE[heading][..., None])
    return np.where(legal.any(-1, keepdims=True), legal, open_)


def greedy_move(nbr, xy, pos, heading, target_rc):
    """Legal move whose next tile has the smallest squared straight-line distance.

    Ties go to the earliest of up, left, down, right (argmin takes the first).
    """
    legal = legal_moves(nbr, pos, heading)
    nxt_rc = xy[np.maximum(nbr[pos], 0)]  # (..., 4, 2)
    d2 = ((nxt_rc - np.asarray(target_rc)[..., None, :]) ** 2).sum(-1)
    return np.argmin(np.where(legal, d2, BIG), axis=-1)
