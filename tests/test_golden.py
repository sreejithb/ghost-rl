"""Golden replays: tripwires for any change to the classic rules or the physics.

If one fails, find out WHY before touching the snapshot. Only regenerate
(tests/make_golden.py) after an intended rule change whose own test was written first.
"""
from pathlib import Path

import numpy as np

from golden_game import FULL_PATH, GOLDEN_PATH, play_actions, play_golden

ROOT = Path(__file__).resolve().parent.parent


def _same(got, want):
    assert len(got["pac_pos"]) == len(want["pac_pos"]), "game length changed"
    for key in want.files:
        assert np.array_equal(got[key], want[key]), f"{key} differs from the golden snapshot"


def test_short_scatter_game_matches_snapshot():
    """Only 29 ticks, all scatter: it pins the scatter targets and the start, nothing more."""
    _same(play_golden(), np.load(ROOT / GOLDEN_PATH))


def test_full_game_matches_snapshot():
    want = np.load(ROOT / FULL_PATH)
    _same(play_actions(want["pac_actions"]), want)


def test_full_game_covers_the_rules_it_is_meant_to_pin_down():
    want = np.load(ROOT / FULL_PATH)
    assert int(want["switched"].sum()) == 7, "not every scatter/chase switch was played"
    assert want["power"].sum() >= 2, "no power pellets eaten"
    assert want["eaten"].sum() >= 2, "no ghosts eaten"
    assert int(want["outcome"]) in (1, 2)  # ended by capture or clear, not by timeout
