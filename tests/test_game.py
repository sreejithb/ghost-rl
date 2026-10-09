import numpy as np
import pytest

from conftest import CORRIDOR, FORK, NO_SCATTER, RING, hand_maze, place, tile
from ghostrl.maze import parse_maze
from ghostrl.maze import DOWN, LEFT, REVERSE, RIGHT, UP
from ghostrl.sim import Sim, SimConfig


def corridor_sim(dot_at=None, cfg=NO_SCATTER):
    text = CORRIDOR if dot_at is None else CORRIDOR.replace("P..........", "P" + "." * 10)
    m = hand_maze(text)
    if dot_at is None:  # no dots at all
        m.dots0[:] = False
    else:
        m.dots0[:] = False
        m.dots0[tile(m, *dot_at)] = True
    return m, Sim(m, cfg)


# --- speed -------------------------------------------------------------------

def test_speed_15_of_16():
    m = parse_maze(RING)
    m.dots0[:] = False
    sim = Sim(m, SimConfig(max_ticks=5000))
    s = sim.reset(1)
    ghost_moves, pac_moves = np.zeros(4, int), 0
    for t in range(160):
        _, info = sim.step(s, np.array([RIGHT if t % 2 == 0 else LEFT]))
        ghost_moves += info["ghost_moved"][0]
        pac_moves += info["pac_moved"][0]
    assert ghost_moves.tolist() == [150] * 4
    assert pac_moves == 160


def test_per_ghost_speed_is_a_parameter():
    m = parse_maze(RING)
    sim = Sim(m, SimConfig(max_ticks=5000, ghost_skip_period=(0, 16, 8, 4)))
    s = sim.reset(1)
    moves = np.zeros(4, int)
    for t in range(160):
        _, info = sim.step(s, np.array([RIGHT if t % 2 == 0 else LEFT]))
        moves += info["ghost_moved"][0]
    assert moves.tolist() == [160, 150, 140, 120]


# --- capture -----------------------------------------------------------------

def test_capture_same_tile():
    m, sim = corridor_sim()
    s = sim.reset(1)
    place(s, m, pac=(1, 6), ghosts={0: (1, 4)}, ghost_dirs={0: RIGHT})
    r, info = sim.step(s, np.array([LEFT]))
    assert tuple(m.xy[s.pac_pos[0]]) == tuple(m.xy[s.ghost_pos[0, 0]]) == (1, 5)
    assert info["captured"][0] and s.done[0]
    assert r[0] == pytest.approx(1 - 0.01)


def test_capture_swap():
    m, sim = corridor_sim()
    s = sim.reset(1)
    place(s, m, pac=(1, 5), ghosts={0: (1, 4)}, ghost_dirs={0: RIGHT})
    _, info = sim.step(s, np.array([LEFT]))
    assert tuple(m.xy[s.pac_pos[0]]) == (1, 4) and tuple(m.xy[s.ghost_pos[0, 0]]) == (1, 5)
    assert info["captured"][0]  # they passed through each other


def test_no_false_capture_when_ghost_follows_one_tile_behind():
    m, sim = corridor_sim()
    s = sim.reset(1)
    place(s, m, pac=(1, 4), pac_dir=RIGHT, ghosts={0: (1, 3)}, ghost_dirs={0: RIGHT})
    _, info = sim.step(s, np.array([RIGHT]))
    assert not info["captured"][0] and not s.done[0]
    assert tuple(m.xy[s.pac_pos[0]]) == (1, 5) and tuple(m.xy[s.ghost_pos[0, 0]]) == (1, 4)


def test_ghost_skip_tick_lets_pacman_walk_onto_it():
    m, sim = corridor_sim(cfg=SimConfig(scatter_ticks=(0, 0, 0, 0)))
    s = sim.reset(1)
    place(s, m, pac=(1, 6), ghosts={0: (1, 5)}, ghost_dirs={0: RIGHT}, tick=15)  # tick 15: all skip
    _, info = sim.step(s, np.array([LEFT]))
    assert not info["ghost_moved"].any()
    assert info["captured"][0]


def test_capture_beats_clearing_on_the_same_tick():
    m, sim = corridor_sim(dot_at=(1, 5))
    s = sim.reset(1)
    place(s, m, pac=(1, 6), ghosts={0: (1, 4)}, ghost_dirs={0: RIGHT})
    r, info = sim.step(s, np.array([LEFT]))
    assert not s.dots.any()
    assert info["captured"][0] and not info["cleared"][0]
    assert r[0] == pytest.approx(0.99)


def test_clear_and_timeout_rewards():
    m, sim = corridor_sim(dot_at=(1, 5))
    s = sim.reset(1)
    place(s, m, pac=(1, 6))
    r, info = sim.step(s, np.array([LEFT]))
    assert info["cleared"][0] and s.done[0] and r[0] == pytest.approx(-1.01)

    m, sim = corridor_sim(cfg=SimConfig(scatter_ticks=(0, 0, 0, 0), max_ticks=5))
    s = sim.reset(1)
    total = 0.0
    for _ in range(5):
        r, info = sim.step(s, np.array([UP]))  # parked
        total += r[0]
    assert info["timeout"][0] and s.done[0]
    assert total == pytest.approx(-0.05)


def test_finished_games_stay_frozen():
    m, sim = corridor_sim()
    s = sim.reset(2)
    s.done[0] = True
    snap = s.copy()
    r, _ = sim.step(s, np.array([LEFT, LEFT]))
    assert r[0] == 0 and r[1] != 0
    assert s.pac_pos[0] == snap.pac_pos[0] and s.tick[0] == snap.tick[0]
    assert (s.ghost_pos[0] == snap.ghost_pos[0]).all()


def test_illegal_pacman_move_is_ignored():
    m, sim = corridor_sim()
    s = sim.reset(1)
    _, info = sim.step(s, np.array([UP]))
    assert not info["pac_moved"][0] and s.pac_pos[0] == m.pac_start


# --- the controller seam ------------------------------------------------------

def test_supplied_ghost_actions_are_used_in_chase_only():
    m = hand_maze(FORK)
    for cfg, expected in ((NO_SCATTER, (1, 3)), (SimConfig(), (1, 5))):
        sim = Sim(m, cfg)
        s = sim.reset(1)
        place(s, m, pac=(4, 5), ghosts={0: (1, 4)}, ghost_dirs={0: DOWN})
        acts = sim.classic_actions(s)
        acts[0, 0] = LEFT
        sim.step(s, np.array([UP]), acts)
        # chase: the supplied LEFT is taken (classic would go right);
        # scatter (default clock, tick 0): scripted, so the supplied action is ignored.
        assert tuple(m.xy[s.ghost_pos[0, 0]]) == expected


def test_illegal_ghost_action_is_rejected():
    m = hand_maze(FORK)
    sim = Sim(m, NO_SCATTER)
    s = sim.reset(1)
    place(s, m, pac=(4, 5), ghosts={0: (1, 4)}, ghost_dirs={0: DOWN})
    acts = sim.classic_actions(s)
    for bad in (UP, DOWN):  # UP reverses the heading, DOWN is a wall
        acts[0, 0] = bad
        with pytest.raises(ValueError):
            sim.step(s.copy(), np.array([UP]), acts)


# --- batching, determinism, invariants ------------------------------------------

def random_legal(sim, state, rng):
    legal = sim.pac_legal(state)
    return np.argmax(rng.random(legal.shape) * legal, axis=1)


def run_games(sim, B, ticks, seed):
    rng = np.random.default_rng(seed)
    s = sim.reset(B)
    for _ in range(ticks):
        sim.step(s, random_legal(sim, s, rng))
    return s


def test_batching_equals_single_games(arcade):
    sim = Sim(arcade)
    B, T = 64, 250
    rng = np.random.default_rng(7)
    acts = rng.random((T, B, 4))
    batch = sim.reset(B)
    singles = [sim.reset(1) for _ in range(B)]
    mixed = False
    for t in range(T):
        sim.step(batch, np.argmax(acts[t] * sim.pac_legal(batch), axis=1))
        for b in range(B):
            sim.step(singles[b], np.argmax(acts[t, b:b + 1] * sim.pac_legal(singles[b]), axis=1))
            for k, v in batch.__dict__.items():
                assert (v[b] == singles[b].__dict__[k][0]).all(), (t, b, k)
        mixed |= batch.done.any() and not batch.done.all()
    assert mixed  # the test covered live and finished games side by side


def test_determinism(arcade):
    sim = Sim(arcade)
    a, b = run_games(sim, 4, 300, seed=1), run_games(sim, 4, 300, seed=1)
    c = run_games(sim, 4, 300, seed=2)
    assert a.hash() == b.hash()
    assert a.hash() != c.hash()


def test_invariants_under_random_play(arcade):
    sim = Sim(arcade)
    m = arcade
    B, ticks = 8, 10_000
    rng = np.random.default_rng(0)
    s = sim.reset(B)
    switch = set(sim.switch_ticks().tolist())
    finished = 0
    for _ in range(ticks):
        was_active = ~s.done
        t_before = s.tick.copy()
        pac0, g0, gdir0, dots0 = s.pac_pos.copy(), s.ghost_pos.copy(), s.ghost_dir.copy(), s.dots.copy()
        _, info = sim.step(s, random_legal(sim, s, rng))

        assert (m.grid[tuple(m.xy[s.pac_pos].T)] != "#").all()
        assert (m.grid[tuple(m.xy[s.ghost_pos].reshape(-1, 2).T)] != "#").all()
        # every move goes to a neighbour (or stays still)
        assert ((s.pac_pos == pac0) | (m.nbr[pac0] == s.pac_pos[:, None]).any(1)).all()
        step_dir = (m.nbr[g0] == s.ghost_pos[..., None]).argmax(-1)
        went = (s.ghost_pos != g0) & ~info["ghosts_eaten"]  # an eaten ghost jumps back to its start
        assert ((m.nbr[g0] == s.ghost_pos[..., None]).any(-1) | ~went | info["ghosts_eaten"]).all()
        assert (went == (info["ghost_moved"] & ~info["ghosts_eaten"])).all()
        # no reversal except on a mode switch or right after a power pellet
        rev = went & (step_dir == REVERSE[gdir0])
        for b in np.flatnonzero(rev.any(1)):
            assert was_active[b] and info["forced_reverse"][b][rev[b]].all()
        # dots only decrease
        assert not (s.dots & ~dots0)[was_active].any()

        finished += int((s.done & was_active).sum())
        sim.reset_where(s, s.done)
    assert finished > 20  # many games started and ended during the fuzz
