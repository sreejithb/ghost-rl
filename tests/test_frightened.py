"""Frightened mode: a power pellet turns every ghost scripted (random walk, half speed) for a while.

Both teams get the same scripted behaviour; the learned policy never chooses in this mode.
"""
import numpy as np
import pytest

from conftest import NO_SCATTER, RING, hand_maze, place, tile
from ghostrl.maze import DOWN, LEFT, REVERSE, RIGHT, UP, parse_maze
from ghostrl.sim import Sim, SimConfig

F = 45
POWER = "#############\n#P.o........#\n#############\n"  # pellet two tiles right of Pac-Man
CFG = SimConfig(scatter_ticks=(0, 0, 0, 0), fright_ticks=F)


def power_sim(cfg=CFG, text=POWER):
    m = hand_maze(text)
    m.dots0[:] = False
    m.dots0[tile(m, 1, 3)] = True  # the pellet
    m.dots0[tile(m, 1, 11)] = True  # a far dot so eating the pellet never clears the maze
    return m, Sim(m, cfg)


def eat_pellet(sim, s, m):
    place(s, m, pac=(1, 2), pac_dir=RIGHT)
    return sim.step(s, np.array([RIGHT]))


def test_maze_marks_exactly_the_energizers(arcade, small):
    assert arcade.power0.sum() == 4 and small.power0.sum() == 4
    assert arcade.dots0[arcade.power0].all()  # they are also dots
    assert arcade.dots0.sum() == 244


def test_eating_a_power_pellet_frightens_every_ghost_and_reverses_them():
    m, sim = power_sim()
    s = sim.reset(1)
    place(s, m, ghosts={0: (1, 9), 1: (1, 8), 2: (1, 10), 3: (1, 11)},
          ghost_dirs={0: LEFT, 1: LEFT, 2: RIGHT, 3: RIGHT})
    _, info = eat_pellet(sim, s, m)
    assert info["power"][0]
    assert s.frightened[0].all() and s.fright_left[0] == F


def test_ordinary_dots_do_not_frighten():
    m, sim = power_sim()
    m.power0[:] = False
    s = sim.reset(1)
    _, info = eat_pellet(sim, s, m)
    assert not info["power"][0] and not s.frightened.any() and s.fright_left[0] == 0


def test_frightened_ghosts_reverse_when_the_pellet_is_eaten():
    m, sim = power_sim()
    s = sim.reset(1)
    place(s, m, ghosts={0: (1, 9)}, ghost_dirs={0: LEFT})
    eat_pellet(sim, s, m)
    assert s.ghost_dir[0, 0] == RIGHT and s.ghost_rev[0, 0]


def test_touching_a_frightened_ghost_eats_it_instead_of_being_caught():
    m, sim = power_sim()
    s = sim.reset(1)
    place(s, m, pac=(1, 2), pac_dir=RIGHT, ghosts={0: (1, 4)}, ghost_dirs={0: LEFT})
    r, info = sim.step(s, np.array([RIGHT]))  # eats the pellet at (1,3), ghost steps onto it
    assert info["ghosts_eaten"][0].tolist() == [True, False, False, False]
    assert not info["captured"][0] and not s.done[0]
    assert s.ghost_pos[0, 0] == m.ghost_start[0] and not s.frightened[0, 0]
    assert s.frightened[0, 1:].all()  # the others are still frightened
    assert r[0] == pytest.approx(-0.01 + sim.cfg.reward_ghost_eaten)


def test_a_normal_ghost_still_catches_pacman_when_another_is_eaten_the_same_tick():
    m, sim = power_sim(cfg=SimConfig(scatter_ticks=(0, 0, 0, 0), fright_ticks=F, power_pellets=False))
    s = sim.reset(1)
    place(s, m, pac=(1, 5), ghosts={0: (1, 4)}, ghost_dirs={0: RIGHT})
    _, info = sim.step(s, np.array([LEFT]))
    assert info["captured"][0] and not info["ghosts_eaten"].any()


def test_frightened_ends_after_the_configured_ticks():
    m, sim = power_sim()
    s = sim.reset(1)
    eat_pellet(sim, s, m)
    for _ in range(F - 1):
        sim.step(s, np.array([LEFT if s.tick[0] % 2 else RIGHT]))
        assert s.frightened.all()
    sim.step(s, np.array([LEFT if s.tick[0] % 2 else RIGHT]))
    assert not s.frightened.any() and s.fright_left[0] == 0


def test_frightened_ghosts_move_at_half_speed():
    m = parse_maze(RING)
    m.dots0[:] = False
    m.power0[:] = False
    sim = Sim(m, SimConfig(scatter_ticks=(0, 0, 0, 0), max_ticks=5000, fright_ticks=40))
    s = sim.reset(1)
    s.frightened[:] = True
    s.fright_left[:] = 40
    moves = np.zeros(4, int)
    for t in range(40):
        _, info = sim.step(s, np.array([RIGHT if t % 2 == 0 else LEFT]))
        moves += info["ghost_moved"][0]
    assert moves.tolist() == [20] * 4


def test_frightened_movement_is_scripted_legal_and_ignores_supplied_actions():
    m = parse_maze(RING)
    m.dots0[:] = False
    sim = Sim(m, SimConfig(scatter_ticks=(0, 0, 0, 0), max_ticks=5000, fright_ticks=60))
    s = sim.reset(1)
    s.frightened[:] = True
    s.fright_left[:] = 60
    bogus = np.full((1, 4), UP)  # a wall for most of these tiles; must not be used
    for t in range(60):
        prev_pos, prev_dir = s.ghost_pos.copy(), s.ghost_dir.copy()
        _, info = sim.step(s, np.array([RIGHT if t % 2 == 0 else LEFT]), bogus)
        moved = info["ghost_moved"][0]
        assert (m.nbr[prev_pos[0], s.ghost_dir[0]][moved] == s.ghost_pos[0][moved]).all()
        assert (s.ghost_dir[0][moved] != REVERSE[prev_dir[0]][moved]).all() or info["forced_reverse"].any()


def test_frightened_walk_is_deterministic_and_actually_varies(arcade):
    def run():
        sim = Sim(arcade, SimConfig(scatter_ticks=(0, 0, 0, 0), fright_ticks=200))
        s = sim.reset(1)
        s.frightened[:] = True
        s.fright_left[:] = 200
        trail = []
        for t in range(120):
            sim.step(s, np.array([LEFT if t % 2 else RIGHT]))
            trail.append(s.ghost_pos[0].copy())
        return np.array(trail)

    a, b = run(), run()
    assert np.array_equal(a, b)
    assert len({tuple(r) for r in a}) > 20  # not stuck


def test_mode_clock_pauses_while_frightened():
    m, sim = power_sim(cfg=SimConfig(scatter_ticks=(10, 10, 10, 10), chase_ticks=(10, 10, 10), fright_ticks=F))
    s = sim.reset(1)
    s.tick[0] = 5  # scatter until 10
    s.ghost_pos[0] = m.ghost_start  # parked
    switched = []
    for t in range(80):
        place(s, m, pac=(1, 2), pac_dir=RIGHT) if t == 0 else None
        _, info = sim.step(s, np.array([RIGHT if t == 0 else (LEFT if t % 2 else RIGHT)]))
        if info["switched"][0]:
            switched.append(int(s.tick[0]) - 1)
        place(s, m, pac=(1, 1)) if t == 0 else None
    # Without the pellet the first switch would be at tick 10; the F frightened ticks push it back.
    assert switched[0] >= 10 + F


def test_power_pellets_can_be_turned_off():
    m, sim = power_sim(cfg=SimConfig(scatter_ticks=(0, 0, 0, 0), fright_ticks=F, power_pellets=False))
    s = sim.reset(1)
    _, info = eat_pellet(sim, s, m)
    assert not info["power"][0] and not s.frightened.any()
