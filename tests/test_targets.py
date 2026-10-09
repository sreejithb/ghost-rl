import numpy as np

from ghostrl import classic
from ghostrl.maze import DOWN, LEFT, RIGHT, UP

CORNER = np.array([30, 0])


def rc(*a):
    return np.array(a)


def test_pinky_facing_right():
    assert classic.pinky_target(rc(10, 10), RIGHT).tolist() == [10, 14]


def test_pinky_facing_up_has_the_left_offset_bug():
    assert classic.pinky_target(rc(10, 10), UP).tolist() == [6, 6]


def test_pinky_other_directions_have_no_bug():
    assert classic.pinky_target(rc(10, 10), LEFT).tolist() == [10, 6]
    assert classic.pinky_target(rc(10, 10), DOWN).tolist() == [14, 10]


def test_inky_facing_right():
    # pivot (10, 12); target = pivot + (pivot - Blinky)
    assert classic.inky_target(rc(10, 10), RIGHT, rc(10, 5)).tolist() == [10, 19]
    assert classic.pinky_target(rc(10, 10), RIGHT, offset=2).tolist() == [10, 12]


def test_inky_facing_up():
    assert classic.pinky_target(rc(10, 10), UP, offset=2).tolist() == [8, 8]
    assert classic.inky_target(rc(10, 10), UP, rc(14, 12)).tolist() == [2, 4]


def test_clyde_far_chases():
    assert classic.clyde_target(rc(10, 10), rc(18, 10), CORNER).tolist() == [10, 10]  # offset (8, 0), d2 = 64


def test_clyde_near_goes_home():
    assert classic.clyde_target(rc(10, 10), rc(15, 16), CORNER).tolist() == CORNER.tolist()  # d2 = 61


def test_offsets_are_parameters():
    assert classic.pinky_target(rc(10, 10), RIGHT, offset=3).tolist() == [10, 13]
    assert classic.inky_target(rc(10, 10), RIGHT, rc(10, 5), pivot=3).tolist() == [10, 21]
    assert classic.clyde_target(rc(10, 10), rc(15, 16), CORNER, radius_sq=61).tolist() == [10, 10]


def test_chase_targets_batch_matches_scalars():
    corners = np.array([[-3, 25], [-3, 2], [31, 27], [31, 0]])
    pac = np.array([[10, 10], [3, 4]])
    pdir = np.array([UP, RIGHT])
    ghosts = np.array([[[14, 12], [0, 0], [0, 0], [18, 10]], [[3, 9], [1, 1], [2, 2], [4, 5]]])
    t = classic.chase_targets(pac, pdir, ghosts, corners)
    assert t.shape == (2, 4, 2)
    assert t[0, 0].tolist() == [10, 10]
    assert t[0, 1].tolist() == [6, 6]
    assert t[0, 2].tolist() == [2, 4]
    assert t[0, 3].tolist() == [10, 10]
    assert t[1, 3].tolist() == [31, 0]  # Clyde at d2 = 2 goes home
