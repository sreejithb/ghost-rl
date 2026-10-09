"""Golden replay: a tripwire for any change to the classic rules or the physics.

If this fails, find out WHY before touching the snapshot. Only regenerate
(tests/make_golden.py) after an intended rule change whose own test was written first.
"""
from pathlib import Path

import numpy as np

from golden_game import GOLDEN_PATH, play_golden

ROOT = Path(__file__).resolve().parent.parent


def test_golden_replay_matches_snapshot():
    want = np.load(ROOT / GOLDEN_PATH)
    got = play_golden()
    assert len(got["pac_pos"]) == len(want["pac_pos"]), "game length changed"
    for key in want.files:
        assert np.array_equal(got[key], want[key]), f"{key} differs from the golden snapshot"


def test_golden_game_is_a_real_full_game():
    want = np.load(ROOT / GOLDEN_PATH)
    assert len(want["pac_pos"]) >= 100
    assert int(want["outcome"]) in (1, 2)  # ended by capture or clear, not by timeout
