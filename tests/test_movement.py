import numpy as np

from conftest import FORK, NO_SCATTER, PLUS, RING, hand_maze, place, tile
from ghostrl import classic
from ghostrl.maze import DOWN, LEFT, RIGHT, UP
from ghostrl.sim import Sim

CHASE = NO_SCATTER


def choose(maze, pos_rc, heading, target):
    pos = tile(maze, *pos_rc)
    return int(classic.greedy_move(maze.nbr, maze.xy, np.array(pos), np.array(heading), np.array(target)))


def test_never_reverses_with_target_directly_behind():
    m = hand_maze("#######\n#.....#\n#.###.#\n#.....#\n#######\n#P....#\n#######\n")
    # At (1,3) heading right the target (1,1) is two tiles behind. Going on is the only legal move.
    assert choose(m, (1, 3), RIGHT, (1, 1)) == RIGHT
    # At the ring's corner (1,5) heading right it must turn down, never back left.
    assert choose(m, (1, 5), RIGHT, (1, 0)) == DOWN


def test_reverse_is_masked_even_when_it_is_the_best_move():
    m = hand_maze(PLUS)
    # Heading right out of the centre: left is the reverse, and is exactly on the target.
    legal = classic.legal_moves(m.nbr, np.array(tile(m, 2, 3)), np.array(RIGHT))
    assert not legal[LEFT]
    assert choose(m, (2, 3), RIGHT, (2, 2)) in (UP, DOWN)  # not LEFT


def test_ties_break_up_left_down_right():
    m = hand_maze(PLUS)  # centre (2,3); arms (1,3) (2,2) (3,3) (2,4)
    assert choose(m, (2, 3), UP, (1, 2)) == UP        # up vs left tie
    assert choose(m, (2, 3), DOWN, (3, 2)) == LEFT    # left vs down tie
    assert choose(m, (2, 3), RIGHT, (3, 4)) == DOWN   # down vs right tie
    assert choose(m, (2, 3), UP, (1, 4)) == UP        # up vs right tie


def test_straight_line_distance_is_squared_euclidean():
    m = hand_maze(PLUS)
    # Target (0, 6): up tile (1,3) is 1+9=10 away, right tile (2,4) is 4+4=8 away.
    assert choose(m, (2, 3), UP, (0, 6)) == RIGHT


def test_dead_end_allows_reversal():
    m = hand_maze("######\n#...P#\n######\n")
    assert choose(m, (1, 4), RIGHT, (1, 1)) == LEFT


def test_short_sighted_takes_the_long_way():
    """At the fork, left is the shorter path to Pac-Man but right is straight-line closer."""
    m = hand_maze(FORK)
    pac = tile(m, 4, 5)
    left, right = tile(m, 1, 3), tile(m, 1, 5)
    assert m.dist_free[left, pac] < m.dist_free[right, pac]  # the real shortest way is left
    assert choose(m, (1, 4), DOWN, (4, 5)) == RIGHT          # the ghost still goes right


def test_short_sighted_ghost_still_arrives_by_the_long_way():
    m = hand_maze(FORK)
    sim = Sim(m, CHASE)
    s = sim.reset(1)
    # Only Blinky enters the fork; the others stay in their sealed corridor.
    place(s, m, pac=(4, 5), ghosts={0: (1, 4)}, ghost_dirs={0: DOWN})
    info = {}
    for _ in range(80):
        _, info = sim.step(s, np.array([UP]))  # Pac-Man bumps the wall and stays put
        if s.done[0]:
            break
    assert info["captured"][0]
    shortest = m.dist_free[tile(m, 1, 4), tile(m, 4, 5)]
    assert shortest == 10
    assert s.tick[0] >= 12  # took the 12-step way round, not the 10-step one
