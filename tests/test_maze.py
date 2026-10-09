import numpy as np

from ghostrl.maze import DOWN, LEFT, REVERSE, RIGHT, UP, parse_maze


def test_arcade_shape_and_dots(arcade):
    assert arcade.grid.shape == (31, 28)
    assert arcade.dots0.sum() == 244  # 240 dots + 4 energizers


def test_small_maze_loads(small):
    assert small.dots0.sum() == sum(c in ".o" for c in small.grid.ravel())
    assert (small.nbr >= 0).sum(1).min() >= 2  # no dead ends


def test_neighbours_symmetric(arcade, small):
    for m in (arcade, small):
        for d in range(4):
            has = m.nbr[:, d] >= 0
            back = m.nbr[m.nbr[has, d], REVERSE[d]]
            assert (back == np.flatnonzero(has)).all()


def test_tunnel_wraps_both_ways(arcade):
    left_mouth, right_mouth = arcade.tile_id[14, 0], arcade.tile_id[14, 27]
    assert left_mouth >= 0 and right_mouth >= 0
    assert arcade.nbr[left_mouth, LEFT] == right_mouth
    assert arcade.nbr[right_mouth, RIGHT] == left_mouth


def test_arcade_has_no_dead_ends(arcade):
    assert (arcade.nbr >= 0).sum(1).min() >= 2


def test_arcade_is_connected(arcade):
    assert (arcade.dist_free < 32767).all()


def test_starts_and_corners(arcade):
    assert tuple(arcade.xy[arcade.pac_start]) == (23, 13)
    assert len(set(arcade.ghost_start)) == 4
    assert arcade.corners.tolist() == [[-3, 25], [-3, 2], [31, 27], [31, 0]]


def test_directions_are_up_left_down_right():
    m = parse_maze("#####\n#.G.#\n#GPG#\n#.G.#\n#####\n")
    t = m.tile_id[2, 2]
    assert tuple(m.xy[m.nbr[t, UP]]) == (1, 2)
    assert tuple(m.xy[m.nbr[t, LEFT]]) == (2, 1)
    assert tuple(m.xy[m.nbr[t, DOWN]]) == (3, 2)
    assert tuple(m.xy[m.nbr[t, RIGHT]]) == (2, 3)


def test_dist_ghost_respects_no_reverse():
    from conftest import hand_maze
    m = hand_maze("#######\n#.....#\n#.###.#\n#.....#\n#######\n#P....#\n#######\n")
    t = m.tile_id[1, 2]
    # Heading right, the tile behind (1,1) is only reachable by going all the way round.
    heading_right = m.dist_ghost[t * 4 + RIGHT, m.tile_id[1, 1]]
    heading_left = m.dist_ghost[t * 4 + LEFT, m.tile_id[1, 1]]
    assert heading_left == 1
    assert heading_right > 1
