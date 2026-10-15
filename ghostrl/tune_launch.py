"""Start the stage-1 tuning jobs on the deployed Modal app. They keep running if this process or your
network dies; results land in the Volume "ghost-rl-runs" under tune/<stage>/ and `collect` pulls them.

    modal deploy ghostrl/modal_app.py
    python -m ghostrl.tune_launch stage1
    python -m ghostrl.tune_launch collect stage1
"""
import json
import subprocess
import sys
import time
from pathlib import Path

from .tune import BASELINE, sample_configs


def launch_stage1(n=24, games=400, seed=0):
    import modal

    f = modal.Function.from_name("ghost-rl", "tune_remote")
    configs = [BASELINE] + sample_configs(n, seed)
    for c in configs:
        f.spawn(c, games, seed + 1, "classic", "stage1")
    print(f"spawned {len(configs)} jobs ({games} games each)")


def launch_confirm(top=3, games=1000, seed=2):
    """Baseline, the `top` configs by clear rate (ties: fewer dots left) and the best by return."""
    import modal

    res = [r for r in load("stage1") if r["cfg"] != BASELINE]
    by_clear = sorted(res, key=lambda r: (-r["clears"], r["dots_left"]))[:top]
    best_ret = max(res, key=lambda r: r["ret"])
    picks = [r["cfg"] for r in by_clear] + ([best_ret["cfg"]] if best_ret["cfg"] not in [r["cfg"] for r in by_clear] else [])
    f = modal.Function.from_name("ghost-rl", "tune_remote")
    for c in [BASELINE] + picks:
        f.spawn(c, games, seed, "classic", "stage2")
    print(f"spawned stage 2: baseline + {len(picks)} configs x {games} games")
    return picks


def launch_other(cfg, games=300, seed=3):
    import modal

    f = modal.Function.from_name("ghost-rl", "tune_remote")
    for c in (BASELINE, cfg):
        for g in ("chaser", "random"):
            f.spawn(c, games, seed, g, "stage3")
    print("spawned stage 3: baseline and best vs path-chasing and random ghosts")


def load(stage):
    return [json.load(open(p)) for p in sorted(Path("runs/tune", stage).glob("*.json"))]


def collect(stage, expect):
    dest = Path("runs/tune")
    dest.mkdir(parents=True, exist_ok=True)
    while True:
        r = subprocess.run(["modal", "volume", "get", "--force", "ghost-rl-runs", f"tune/{stage}", str(dest)],
                           capture_output=True, text=True)
        n = len(list(Path("runs/tune", stage).glob("*.json"))) if Path("runs/tune", stage).exists() else 0
        print(time.strftime("%H:%M:%S"), f"{n}/{expect} results", flush=True)
        if n >= expect:
            return
        time.sleep(30)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "stage1":
        launch_stage1()
    elif cmd == "confirm":
        launch_confirm()
    elif cmd == "collect":
        collect(sys.argv[2], int(sys.argv[3]))
