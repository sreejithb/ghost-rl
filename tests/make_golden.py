"""Regenerate the golden snapshot. Run by hand, only when a rule change is intended
and its failing test is already written:  python tests/make_golden.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np

from golden_game import GOLDEN_PATH, play_golden

g = play_golden()
Path(GOLDEN_PATH).parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(GOLDEN_PATH, **g)
print(f"saved {GOLDEN_PATH}: {len(g['pac_pos'])} ticks, outcome {int(g['outcome'])}")
