"""GIF export. A game is a pac action log (plus optional ghost actions); replay it, draw each tick."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw

from .maze import GHOST_NAMES
from .sim import Sim

GHOST_RGB = {"blinky": (255, 0, 0), "pinky": (255, 184, 255), "inky": (0, 255, 255), "clyde": (255, 184, 82)}
WALL, WALL_EDGE, DOT, PAC = (20, 20, 150), (60, 60, 255), (255, 210, 170), (255, 235, 0)
SCATTER_BANNER, CHASE_BANNER, FRIGHT_BANNER = (60, 140, 255), (255, 70, 70), (40, 40, 220)
FRIGHT_RGB, FRIGHT_FLASH = (33, 33, 222), (255, 255, 255)
HUD = 22


def replay(sim: Sim, pac_actions, ghost_actions=None):
    """Yield (state snapshot, info) per tick for a single game, starting with the initial state."""
    s = sim.reset(1)
    yield s.copy(), None
    for t, a in enumerate(pac_actions):
        g = None if ghost_actions is None else np.asarray(ghost_actions[t])[None]
        _, info = sim.step(s, np.array([a]), g)
        yield s.copy(), info
        if s.done[0]:
            break


def draw_frame(sim: Sim, s, scale: int = 16, outcome: str = "") -> Image.Image:
    m = sim.maze
    W, H = m.W * scale, m.H * scale
    img = Image.new("RGB", (W, H + HUD), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, HUD], fill=(0, 0, 0))
    scatter = bool(sim.is_scatter(sim.mode_time(s))[0])
    fright = int(s.fright_left[0])
    banner, label = (FRIGHT_BANNER, "FRIGHTENED") if fright > 0 else (
        (SCATTER_BANNER, "SCATTER") if scatter else (CHASE_BANNER, "CHASE"))
    d.rectangle([0, 0, 92, HUD - 4], fill=banner)
    d.text((6, 4), label, fill=(255, 255, 255) if fright > 0 else (0, 0, 0))
    d.text((100, 4), f"tick {int(s.tick[0])}   dots left {int(s.dots[0].sum())}   {outcome}", fill=(255, 255, 255))

    oy = HUD
    for r in range(m.H):
        for c in range(m.W):
            if m.grid[r, c] == "#":
                x, y = c * scale, r * scale + oy
                d.rectangle([x, y, x + scale - 1, y + scale - 1], fill=WALL)
                for dr, dc, box in ((-1, 0, (0, 0, scale - 1, 1)), (1, 0, (0, scale - 2, scale - 1, scale - 1)),
                                    (0, -1, (0, 0, 1, scale - 1)), (0, 1, (scale - 2, 0, scale - 1, scale - 1))):
                    rr, cc = r + dr, c + dc
                    if not (0 <= rr < m.H and 0 <= cc < m.W) or m.grid[rr, cc] != "#":
                        d.rectangle([x + box[0], y + box[1], x + box[2], y + box[3]], fill=WALL_EDGE)
    # Energizer tiles are plain dots in v1; they are drawn bigger only so the maze looks familiar.
    for t in np.flatnonzero(s.dots[0]):
        r, c = m.xy[t]
        cx, cy = c * scale + scale / 2, r * scale + scale / 2 + oy
        rad = scale * (0.3 if m.grid[r, c] == "o" else 0.11)
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], fill=DOT)

    # scatter corners: a small ghost-coloured marker on the maze edge nearest each target
    if scatter and fright == 0:
        for g, name in enumerate(GHOST_NAMES):
            r, c = m.corners[g]
            r, c = min(max(r, 0), m.H - 1), min(max(c, 0), m.W - 1)
            x, y = c * scale + scale / 2, r * scale + scale / 2 + oy
            d.polygon([(x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)], outline=GHOST_RGB[name])

    r, c = m.xy[s.pac_pos[0]]
    cx, cy = c * scale + scale / 2, r * scale + scale / 2 + oy
    rad = scale * 0.46
    face = {0: 270, 1: 180, 2: 90, 3: 0}[int(s.pac_dir[0])]  # PIL angles run clockwise from east
    mouth = 35 if s.tick[0] % 2 == 0 else 8
    d.pieslice([cx - rad, cy - rad, cx + rad, cy + rad], face + mouth, face - mouth + 360, fill=PAC)

    for g, name in enumerate(GHOST_NAMES):
        r, c = m.xy[s.ghost_pos[0, g]]
        cx, cy = c * scale + scale / 2, r * scale + scale / 2 + oy
        rad, col = scale * 0.46, GHOST_RGB[name]
        if s.frightened[0, g]:  # blue, flashing white for the last 10 ticks
            col = FRIGHT_FLASH if (fright <= 10 and fright % 2 == 0) else FRIGHT_RGB
        d.pieslice([cx - rad, cy - rad, cx + rad, cy + rad], 180, 360, fill=col)
        d.rectangle([cx - rad, cy, cx + rad, cy + rad * 0.85], fill=col)
        dr, dc = ((-1, 0), (0, -1), (1, 0), (0, 1))[int(s.ghost_dir[0, g])]
        for ex in (-0.32, 0.32):
            ex_, ey_ = cx + ex * scale, cy - 0.1 * scale
            d.ellipse([ex_ - 3, ey_ - 3.5, ex_ + 3, ey_ + 3.5], fill=(255, 255, 255))
            px, py = ex_ + dc * 1.8, ey_ + dr * 1.8
            d.ellipse([px - 1.5, py - 1.5, px + 1.5, py + 1.5], fill=(30, 30, 200))
    return img


def save_gif(sim: Sim, pac_actions, path, ghost_actions=None, scale: int = 16, fps: int = 15,
             hold_end: int = 20) -> int:
    """Replay the logs and write a GIF. Returns the number of ticks drawn."""
    frames, last = [], None
    for s, info in replay(sim, pac_actions, ghost_actions):
        out = ""
        if info is not None:
            out = "CAUGHT" if info["captured"][0] else "CLEARED" if info["cleared"][0] else "TIMEOUT" if info["timeout"][0] else ""
        frames.append(draw_frame(sim, s, scale, out))
        last = out
    frames += [frames[-1]] * hold_end
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=False)
    return len(frames) - hold_end - 1
