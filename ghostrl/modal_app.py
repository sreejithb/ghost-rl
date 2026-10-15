"""Run the Pac-Man PPO job on Modal and watch it in TensorBoard.

    modal run ghostrl/modal_app.py::main                            # CPU smoke test of the environment
    modal run --detach ghostrl/modal_app.py::train --name arcade1    # the full run (keeps going if you close the terminal)
    python -m ghostrl.sync_tb arcade1                                # in another terminal: pull logs, then open TensorBoard
    modal run ghostrl/modal_app.py::stop --name arcade1              # end the run early (it still saves and evaluates)

GPU type: GHOSTRL_GPU=L4 modal run ... (default T4). Everything the run writes lands in the Volume
"ghost-rl-runs" under /<name>/ : tb/ (TensorBoard), metrics.csv, checkpoint.pt, eval.json, config.json.
"""
import os
import time

import modal

app = modal.App("ghost-rl")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "torch", "tensorboard")
    .add_local_python_source("ghostrl")
    .add_local_dir("mazes", remote_path="/root/mazes")
)
runs = modal.Volume.from_name("ghost-rl-runs", create_if_missing=True)
flags = modal.Dict.from_name("ghost-rl-stop", create_if_missing=True)  # name -> True asks that run to stop
GPU = os.environ.get("GHOSTRL_GPU", "T4")


@app.function(image=image, cpu=2, timeout=600)
def env_speed(B: int = 256, steps: int = 200) -> dict:
    import numpy as np

    from ghostrl.env import GhostRLEnv
    from ghostrl.maze import load_maze
    from ghostrl.sim import Sim

    rng = np.random.default_rng(0)
    env = GhostRLEnv(Sim(load_maze("arcade")), B)
    obs = env.reset()
    t0 = time.time()
    for _ in range(steps):
        obs, _, _, _ = env.step((obs["mask"] * rng.random((B, 4))).argmax(1))
    return {"env_steps_per_s": round(B * steps / (time.time() - t0))}


@app.function(image=image, gpu=GPU, cpu=4, volumes={"/runs": runs}, timeout=24 * 3600)
def train_remote(name: str, cfg: dict) -> dict:
    from pathlib import Path

    import torch

    from ghostrl.train import Config, train

    cfg = dict(cfg, out=f"/runs/{name}", device="cuda" if torch.cuda.is_available() else "cpu")
    print("device", cfg["device"], torch.cuda.get_device_name(0) if torch.cuda.is_available() else "", flush=True)
    last = {"t": 0.0}

    def on_update(update: int, out: Path):
        if update > 0 and flags.get(name, False) and not (out / "STOP").exists():
            (out / "STOP").write_text("stop requested from modal")
        if update < 0 or time.time() - last["t"] > 20:  # publish logs so they can be pulled while it runs
            runs.commit()
            last["t"] = time.time()

    result = train(Config(**cfg), on_update=on_update)
    runs.commit()
    return result


@app.local_entrypoint()
def main():
    print(env_speed.remote())


@app.local_entrypoint()
def train(name: str = "arcade1", maze: str = "arcade", steps: int = 20_000_000, envs: int = 256,
          num_steps: int = 128, minibatches: int = 8, seed: int = 0, ent_coef: float = 0.01,
          history: bool = False, lr: float = 2.5e-4, lr_final: float = 0.0, gamma: float = 0.99,
          reward: str = "default", reward_scale: float = 0.2, init_from: str = ""):
    flags.pop(name, None)
    cfg = dict(maze=maze, total_steps=steps, num_envs=envs, num_steps=num_steps, minibatches=minibatches, seed=seed,
               ent_coef=ent_coef, history=history, lr=lr, lr_final=lr_final, gamma=gamma, reward=reward, reward_scale=reward_scale, init_from=init_from, eval_games=200, ckpt_every=10)
    print(f"starting '{name}' on {GPU}: {cfg}")
    print(train_remote.remote(name, cfg))


@app.local_entrypoint()
def stop(name: str):
    flags[name] = True
    print(f"asked '{name}' to stop; it will save, evaluate and exit within one update")


@app.function(image=image, cpu=2, memory=4096, volumes={"/runs": runs}, timeout=3 * 3600)
def tune_remote(cfg: dict, games: int, seed: int, ghosts: str = "classic", tag: str = "stage1") -> dict:
    import hashlib
    import json
    from pathlib import Path

    from ghostrl.tune import run_config

    res = run_config(cfg, games=games, seed=seed, ghosts=ghosts)
    key = hashlib.md5(json.dumps([cfg, games, seed, ghosts, tag], sort_keys=True).encode()).hexdigest()[:10]
    out = Path("/runs/tune") / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{key}.json").write_text(json.dumps(res))
    runs.commit()
    return res


def _row(r):
    c = r["cfg"]
    return (f"return {r['ret']:5.1f} ±{r['ret_se']:.1f} | clears {r['clears']:3d}/{r['games']} | dots left {r['dots_left']:5.1f} | "
            f"ticks {r['ticks']:4.0f} | ghosts {r['ghosts_eaten']:.1f} | {r['steps_per_s']:.0f} st/s | "
            f"N={c['n_rollouts']} D={c['depth']} death={c['death_cost']:g} gb={c['ghost_bonus']:g} pull={c['dot_pull']:g} "
            f"fear={c['fear_w']:g}/{c['fear_cap']:g} dot={c['dotnext_w']:g} noise={c['noise']:g} head={c['heading_bonus']:g} res={int(c['reserve'])}")


@app.local_entrypoint()
def tune(n: int = 24, games: int = 400, seed: int = 0, confirm: int = 3, confirm_games: int = 1000, other_games: int = 300):
    import json
    from pathlib import Path

    from ghostrl.tune import BASELINE, sample_configs

    out = Path("runs/tune")
    out.mkdir(parents=True, exist_ok=True)
    configs = [BASELINE] + sample_configs(n, seed)
    print(f"stage 1: {len(configs)} configs x {games} games (config 0 is the current setting)", flush=True)
    res = list(tune_remote.map(configs, kwargs=dict(games=games, seed=seed + 1, tag="stage1")))
    json.dump(res, open(out / "stage1.json", "w"), indent=1)
    order = sorted(range(len(res)), key=lambda i: -res[i]["ret"])
    for i in order:
        print(("*" if i == 0 else " "), _row(res[i]), flush=True)
    top = [i for i in order if i != 0][:confirm]
    finalists = [configs[0]] + [configs[i] for i in top]
    print(f"\nstage 2: baseline + top {confirm} on {confirm_games} fresh games", flush=True)
    res2 = list(tune_remote.map(finalists, kwargs=dict(games=confirm_games, seed=seed + 2, tag="stage2")))
    json.dump(res2, open(out / "stage2.json", "w"), indent=1)
    for r in sorted(res2, key=lambda r: -r["ret"]):
        print(("*" if r["cfg"] == BASELINE else " "), _row(r), flush=True)
    best = max(res2[1:], key=lambda r: r["ret"])
    print("\nstage 3: best config against other ghost types", flush=True)
    jobs = [(c, g) for c in (BASELINE, best["cfg"]) for g in ("chaser", "random")]
    res3 = list(tune_remote.starmap([(c, other_games, seed + 3, g, "stage3") for c, g in jobs]))
    json.dump(res3, open(out / "stage3.json", "w"), indent=1)
    for r in res3:
        print(("baseline" if r["cfg"] == BASELINE else "best    "), r["ghosts"], _row(r), flush=True)
    print("\nbest config:", json.dumps(best["cfg"]))
