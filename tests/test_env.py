import numpy as np
import pytest

from conftest import CORRIDOR, NO_SCATTER, hand_maze, place, tile
from ghostrl.env import N_PLANES, N_SCALARS, PLANES, GhostRLEnv, PacmanReward
from ghostrl.maze import LEFT, RIGHT
from ghostrl.sim import Sim, SimConfig


def corridor_env(controls="pacman", B=1, **kw):
    m = hand_maze(CORRIDOR)
    m.dots0[:] = False
    m.dots0[tile(m, 1, 3)] = True
    m.dots0[tile(m, 1, 11)] = True
    return m, GhostRLEnv(Sim(m, NO_SCATTER), B, controls, **kw)


def test_observation_shapes_and_contents(arcade):
    env = GhostRLEnv(Sim(arcade), 3)
    o = env.observe()
    assert o["planes"].shape == (3, N_PLANES, arcade.H, arcade.W) and len(PLANES) == N_PLANES
    assert o["scalars"].shape == (3, N_SCALARS) and o["mask"].shape == (3, 4)
    p = o["planes"][0]
    assert p[0].sum() == (arcade.grid == "#").sum()
    assert p[1].sum() == 240 and p[2].sum() == 4  # small dots and power pellets
    assert p[3].sum() == 1 and p[4].sum() == 4 and p[5].sum() == 0
    assert p[6:].sum() == 4  # each live ghost marks its heading once
    assert o["scalars"][0, :4].sum() == 1 and o["scalars"][0, 7] == 1.0


def test_pacman_mask_matches_legal_moves():
    m, env = corridor_env()
    place(env.state, m, pac=(1, 1))
    assert env.observe()["mask"][0].tolist() == [False, False, False, True]  # only right


def test_pacman_rewards_for_dot_death_and_clear():
    m, env = corridor_env(pac_reward=PacmanReward(dot=0.1, power=0.5, ghost_eaten=1, death=-5, clear=10, tick=-0.005))
    place(env.state, m, pac=(1, 2), pac_dir=RIGHT)
    _, r, done, _ = env.step([RIGHT])  # eats the dot at (1,3)
    assert r[0] == pytest.approx(0.1 - 0.005) and not done[0]
    m, env = corridor_env()
    place(env.state, m, pac=(1, 6), ghosts={0: (1, 4)}, ghost_dirs={0: RIGHT})
    _, r, done, info = env.step([LEFT])
    assert done[0] and info["final"]["captured"][0] and r[0] == pytest.approx(-5.0 - 0.005)


def test_games_reset_automatically_and_episode_stats_are_reported():
    m, env = corridor_env()
    place(env.state, m, pac=(1, 6), ghosts={0: (1, 4)}, ghost_dirs={0: RIGHT})
    obs, _, done, info = env.step([LEFT])
    assert done[0] and not env.state.done[0]  # fresh game
    assert info["final"]["length"][0] == 1 and env.ep_len[0] == 0
    assert obs["planes"][0, 3].sum() == 1


def test_ghost_side_uses_shared_reward_and_a_pacman_opponent():
    m, env = corridor_env("ghosts", opponent=lambda s: np.full(s.B, LEFT))
    place(env.state, m, pac=(1, 6), ghosts={0: (1, 4)}, ghost_dirs={0: RIGHT})
    o = env.observe()
    assert o["mask"].shape == (1, 4, 4) and o["ghost_id"].shape == (1, 4, 4)
    acts = np.argmax(o["mask"], -1)  # any legal move per ghost
    _, r, done, info = env.step(acts)
    assert done[0] and info["final"]["captured"][0]
    assert r[0] == pytest.approx(1 - 0.01)  # the sim's team reward


def test_both_sides_use_the_same_sim_step():
    """Pac-Man mode with classic opponents equals the plain Sim stepping the same moves."""
    m, env = corridor_env(B=4)
    ref = Sim(m, NO_SCATTER)
    s = ref.reset(4)
    for t in range(40):
        a = np.full(4, RIGHT if t % 2 == 0 else LEFT)
        env.step(a)
        ref.step(s, a)
        if s.done.any():
            break
        assert np.array_equal(env.state.ghost_pos, s.ghost_pos) and np.array_equal(env.state.pac_pos, s.pac_pos)


def test_history_adds_trail_planes_and_scalars_and_clears_them_on_reset():
    m = hand_maze(CORRIDOR)
    m.dots0[:] = False
    m.dots0[tile(m, 1, 11)] = True
    env = GhostRLEnv(Sim(m, NO_SCATTER), 1, history=True)
    assert env.n_planes == N_PLANES + 4 and env.n_scalars == N_SCALARS + 2
    place(env.state, m, pac=(1, 2), pac_dir=RIGHT)
    o = env.observe()
    assert o["planes"][0, N_PLANES:].sum() == 0  # fresh game: no history yet
    env.step([RIGHT])
    env.step([RIGHT])
    o = env.observe()
    cell = lambda r, c: (r, c)
    assert o["planes"][0, N_PLANES + 2][cell(1, 3)] == 1  # Pac-Man one tick ago
    assert o["planes"][0, N_PLANES + 3][cell(1, 2)] == 1  # and two ticks ago
    assert o["planes"][0, N_PLANES + 0].sum() == 4  # four ghosts one tick ago
    place(env.state, m, pac=(1, 6), ghosts={0: (1, 4)}, ghost_dirs={0: RIGHT})
    env.step([LEFT])  # captured -> auto reset
    assert env.observe()["planes"][0, N_PLANES:].sum() == 0
