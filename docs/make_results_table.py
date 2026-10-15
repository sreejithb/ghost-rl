"""Build docs/planner_results.html from the saved RL evaluations (runs/*/eval.json) and the planner
measurements below. Run from the repo root:  python docs/make_results_table.py
"""
import html
import json
from pathlib import Path

import numpy as np

# name, games, clears, dots_left_mean, ticks_mean, return (default reward), speed, note
PLANNERS = [
    ("Flat Monte-Carlo search (8 futures × 25 ticks)", 200, 9, 49, 347, 20.8, "458 steps/s",
     "Best planner. Plays 8 short futures per move in the real simulator; ghosts modelled as path-chasers. Eats 5.4 ghosts per game."),
    ("Flat search + pellet reserve + same random numbers + heading bonus", 100, 7, 50, 375, None, "≈ 500 steps/s",
     "Earlier test of three tweaks together; no gain over the plain version."),
    ("Flat search, pellet reserve range 4", 100, 7, 63, 447, None, "≈ 500 steps/s",
     "Lives longer, eats fewer dots."),
    ("Flat search, 12 futures × 40 ticks", 40, 1, 53, 341, None, "236 steps/s",
     "More futures and a deeper lookahead did not help."),
    ("UCT tree search, 32 iterations", 100, 0, 57, 311, None, "197 steps/s",
     "Tree search with greedy rollouts; worse and slower than the flat search."),
    ("UCT tree search, 64 iterations", 100, 0, 50, 317, None, "87 steps/s", "Slower still, same result."),
    ("Territory planner (safe-area search, exact ghost timing)", 200, 0, 83, 702, 12.6, "1,301 steps/s",
     "The first planner. Survives long but eats less. Eats 3.9 ghosts per game."),
]
TERRITORY_OTHER = [  # territory planner against other ghost types, 100 games each
    ("Classic ghosts", 100, 0, 99, 1, 81, 645),
    ("Path-chasing ghosts (neutral model)", 100, 0, 55, 45, 78, 1335),
    ("Random-move ghosts", 100, 1, 6, 93, 25, 1952),
]
# group label, run-name prefix, settings, return unit
RL = [
    ("abA_base", "lr 2.5e-4 → 0, entropy 0.05, 256 games", "default"),
    ("abB_hist", "same + trail planes", "default"),
    ("lrB_base", "lr 0.01 → 0.001, entropy 0.05, 512 games", "default"),
    ("lrB_hist", "same + trail planes", "default"),
    ("entD", "lr 0.01 → 0.001, entropy 0.1", "default"),
    ("entC", "lr 0.01 → 0.001, entropy 0.25", "default"),
    ("g999", "default reward, discount 0.999, entropy 0.05", "default"),
    ("atari_g099", "Atari points, discount 0.99, no tick cost", "Atari"),
    ("atari_g09", "Atari points, discount 0.9, no tick cost", "Atari"),
    ("noterm", "no reward for capture or clear, tick −0.05", "tick −0.05"),
    ("noterm_cont", "the noterm runs continued to 10M steps", "tick −0.05"),
]


def rl_rows():
    rows = []
    root = Path("runs")
    for prefix, setting, unit in RL:
        ev = []
        for d in sorted(root.glob(f"{prefix}_s[0-9]")):
            try:
                ev.append(json.load(open(d / "eval.json"))["sampled"])
            except Exception:
                pass
        if not ev:
            continue
        dots = np.array([e["mean_dots_left"] for e in ev])
        rows.append(dict(name=f"PPO · {prefix}", setting=setting, seeds=len(ev), games=200 * len(ev),
                         clears=int(round(sum(e["clear_rate"] * e["games"] for e in ev))),
                         dots=dots.mean(), lo=dots.min(), hi=dots.max(),
                         ticks=np.mean([e["mean_length"] for e in ev]),
                         ret=np.mean([e["mean_return"] for e in ev]) if unit == "default" else None, unit=unit))
    ck = root / "long30M_e01_s0" / "eval_from_checkpoint.json"
    if ck.exists():
        e = json.load(open(ck))["sampled"]
        rows.append(dict(name="PPO · long run (stopped at 19M steps)", setting="lr 0.01 → 0.001, entropy 0.1; cancelled early by Modal",
                         seeds=1, games=200, clears=int(round(e["clear_rate"] * 200)), dots=e["mean_dots_left"], lo=e["mean_dots_left"],
                         hi=e["mean_dots_left"], ticks=e["mean_length"], ret=e["mean_return"], unit="default"))
    return rows


def pct(c, n):
    return f"{c} / {n} ({100 * c / n:.1f}%)"


def main():
    rl = rl_rows()
    best_dots = min([p[3] for p in PLANNERS] + [r["dots"] for r in rl])
    out = []
    A = out.append
    A(HEAD.replace("{RL_GAMES:,}", f"{sum(r['games'] for r in rl):,}"))
    A("<h2>Planners (no training)</h2><div class='wrap'><table><thead><tr><th>Planner</th><th class='n'>Games</th><th class='n'>Mazes cleared</th>"
      "<th class='n'>Dots left (of 244)</th><th class='n'>Ticks per game</th><th class='n'>Return*</th><th class='n'>Speed</th><th>Notes</th></tr></thead><tbody>")
    for name, g, c, dots, ticks, ret, speed, note in PLANNERS:
        cls = " class='best'" if dots == best_dots else ""
        A(f"<tr{cls}><td>{html.escape(name)}</td><td class='n'>{g}</td><td class='n'>{pct(c, g)}</td><td class='n'>{dots}</td>"
          f"<td class='n'>{ticks}</td><td class='n'>{'—' if ret is None else f'{ret:.1f}'}</td><td class='n'>{speed}</td><td>{html.escape(note)}</td></tr>")
    A("</tbody></table></div>")
    A("<h2>Reinforcement learning (PPO, 5M steps unless noted, arcade maze)</h2><div class='wrap'><table><thead><tr><th>Run group</th><th>Settings</th>"
      "<th class='n'>Seeds</th><th class='n'>Games</th><th class='n'>Mazes cleared</th><th class='n'>Dots left, average (range over seeds)</th>"
      "<th class='n'>Ticks per game</th><th class='n'>Return*</th></tr></thead><tbody>")
    for r in rl:
        cls = " class='best'" if round(r["dots"]) == round(best_dots) else ""
        rng = f"{r['lo']:.0f}–{r['hi']:.0f}" if r["seeds"] > 1 else "1 run"
        ret = f"{r['ret']:.1f}" if r["ret"] is not None else f"— ({r['unit']})"
        A(f"<tr{cls}><td>{html.escape(r['name'])}</td><td>{html.escape(r['setting'])}</td><td class='n'>{r['seeds']}</td><td class='n'>{r['games']}</td>"
          f"<td class='n'>{pct(r['clears'], r['games'])}</td><td class='n'>{r['dots']:.0f} ({rng})</td><td class='n'>{r['ticks']:.0f}</td><td class='n'>{ret}</td></tr>")
    A("</tbody></table></div>")
    A("<h2>Territory planner against other ghost types (100 games each)</h2><div class='wrap'><table><thead><tr><th>Ghosts</th><th class='n'>Mazes cleared</th>"
      "<th class='n'>Caught</th><th class='n'>Timed out (2,000 ticks)</th><th class='n'>Dots left</th><th class='n'>Ticks per game</th></tr></thead><tbody>")
    for name, g, c, cap, to, dots, ticks in TERRITORY_OTHER:
        A(f"<tr><td>{name}</td><td class='n'>{c} / {g}</td><td class='n'>{cap}</td><td class='n'>{to}</td><td class='n'>{dots}</td><td class='n'>{ticks}</td></tr>")
    A("</tbody></table></div>")
    A(FOOT)
    Path("docs/planner_results.html").write_text("\n".join(out))
    print("wrote docs/planner_results.html:", len(PLANNERS), "planners,", len(rl), "RL groups")


HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pac-Man baselines</title>
<style>
:root{--bg:#fafaf7;--fg:#1d1d1b;--mute:#6b6b66;--line:#dcdcd4;--head:#efefe8;--best:#e6f3e6;--accent:#2a5bd7}
@media (prefers-color-scheme: dark){:root{--bg:#17171a;--fg:#ececea;--mute:#9a9a94;--line:#34343a;--head:#222227;--best:#1f3324;--accent:#8fb0ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1200px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px}p{color:var(--mute);margin:4px 0}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px}
table{border-collapse:collapse;width:100%;min-width:760px}
th,td{padding:8px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}
th{background:var(--head);font-weight:600;font-size:13px}tr:last-child td{border-bottom:0}
.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr.best td{background:var(--best)}
.foot{font-size:13px;color:var(--mute)}.foot li{margin:3px 0}
</style></head><body><main>
<h1>Pac-Man baselines</h1>
<p>Every number is Pac-Man against the classic ghosts on the arcade maze (244 dots), unless a heading says otherwise. Green rows have the fewest dots left. None of the {RL_GAMES:,} games played by the RL policies clears the maze.</p>"""
FOOT = """<ul class="foot">
<li><b>Mazes cleared</b>: games where Pac-Man ate all 244 dots. <b>Dots left</b>: dots remaining when the game ended. <b>Ticks per game</b>: average game length (the limit is 2,000).</li>
<li>*<b>Return</b> uses the default reward: +0.1 per dot, +0.5 extra per power pellet, +1 per ghost eaten, −5 if caught, +10 for clearing, −0.005 per tick. Groups trained on a different reward show a dash, because their returns are in other units.</li>
<li>Planners play 100–200 games with 2% random moves, so repeated games differ. RL numbers come from 200 sampled-move games per seed. Greedy (most-likely-move) evaluation is a single repeated game and is left out.</li>
<li>With 2 to 3 seeds the RL results are noisy: seeds in one group often differ by more than the gaps between groups. Treat gaps under about 30 dots as noise.</li>
<li>Planner and RL runs measured at different times. RL training speed was about 10,000 steps/s on a T4 GPU (laptop CPU: 300–500).</li>
</ul></main></body></html>"""

if __name__ == "__main__":
    main()
