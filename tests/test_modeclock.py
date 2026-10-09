import numpy as np

from conftest import RING, hand_maze, tile
from ghostrl.maze import parse_maze
from ghostrl.maze import REVERSE
from ghostrl.sim import Sim, SimConfig

SWITCHES = [53, 205, 258, 410, 448, 600, 638]


def pac_shuffle(t):
    return np.array([3 if t % 2 == 0 else 1])  # right, left: stays in his own corridor


def test_switch_ticks_and_schedule():
    sim = Sim(parse_maze(RING))
    assert sim.switch_ticks().tolist() == SWITCHES
    t = np.arange(0, 700)
    sc = sim.is_scatter(t)
    assert sc[:53].all() and not sc[53:205].any() and sc[205:258].all()
    assert not sc[258:410].any() and sc[410:448].all() and not sc[448:600].any()
    assert sc[600:638].all() and not sc[638:].any()
    assert not sim.is_scatter(np.array([10**6]))[0]


def test_every_ghost_reverses_on_exactly_the_switch_ticks():
    m = parse_maze(RING)
    sim = Sim(m, SimConfig(max_ticks=5000))
    s = sim.reset(1)
    reversed_at = []
    for t in range(700):
        _, info = sim.step(s, pac_shuffle(t))
        if info["forced_reverse"].any():
            assert info["forced_reverse"].all()  # all four, together
            reversed_at.append(t)
    assert reversed_at == SWITCHES


def test_reversal_really_turns_the_ghost_around():
    m = parse_maze(RING)
    sim = Sim(m, SimConfig(max_ticks=5000))
    s = sim.reset(1)
    for t in range(53):
        sim.step(s, pac_shuffle(t))
    heading, pos = s.ghost_dir[0].copy(), s.ghost_pos[0].copy()
    _, info = sim.step(s, pac_shuffle(53))
    assert (s.ghost_dir[0] == REVERSE[heading]).all()
    assert (s.ghost_pos[0] != pos).all()
    # ...and the move undid the previous step's tile
    assert info["ghost_old"][0].tolist() == pos.tolist()


def test_reversal_survives_a_skipped_tick():
    """If a ghost skips on the switch tick, it still reverses on its next move."""
    m = parse_maze(RING)
    cfg = SimConfig(scatter_ticks=(5, 5, 5, 5), chase_ticks=(5, 5, 5), ghost_skip_period=(5, 5, 5, 5),
                    max_ticks=5000)
    sim = Sim(m, cfg)  # skip when t % 5 == 4; switch at t = 5, 10, ... never coincides, so shift:
    cfg2 = SimConfig(scatter_ticks=(4, 4, 4, 4), chase_ticks=(6, 6, 6), ghost_skip_period=(5, 5, 5, 5),
                     max_ticks=5000)
    sim = Sim(m, cfg2)
    assert sim.switch_ticks()[0] == 4  # 4 % 5 == 4: ghosts skip on the switch tick
    s = sim.reset(1)
    for t in range(4):
        sim.step(s, pac_shuffle(t))
    before = s.ghost_pos[0].copy()
    _, info = sim.step(s, pac_shuffle(4))
    assert not info["ghost_moved"].any() and info["switched"][0]
    _, info = sim.step(s, pac_shuffle(5))
    assert info["forced_reverse"].all() and info["ghost_moved"].all()
    assert not info["switched"][0]


def test_scatter_ghosts_circle_their_own_corners(arcade):
    cfg = SimConfig(scatter_ticks=(10**6,) * 4, max_ticks=10**6)
    sim = Sim(arcade, cfg)
    s = sim.reset(1)
    trace = []
    for _ in range(1600):
        sim.step(s, np.array([0]))  # Pac-Man pushes into a wall and stays parked at his start
        assert not s.done[0]
        trace.append(s.ghost_pos[0].copy())
    trace = np.array(trace)
    for g in range(4):
        loop = set(trace[400:1000, g])
        assert loop == set(trace[1000:, g])  # settled into a cycle
        assert len(loop) <= 40
        pts = arcade.xy[sorted(loop)]
        mean_dist = np.linalg.norm(pts[:, None, :] - arcade.corners[None], axis=2).mean(0)
        assert mean_dist.argmin() == g  # the loop hugs this ghost's own corner


def test_no_scatter_config_never_reverses():
    m = parse_maze(RING)
    sim = Sim(m, SimConfig(scatter_ticks=(0, 0, 0, 0), max_ticks=5000))
    assert sim.switch_ticks().tolist() == []
    s = sim.reset(1)
    assert not sim.is_scatter(s.tick).any()
    for t in range(700):
        _, info = sim.step(s, pac_shuffle(t))
        assert not info["forced_reverse"].any()
