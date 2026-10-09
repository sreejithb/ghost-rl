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
          history: bool = False, lr: float = 2.5e-4, lr_final: float = 0.0):
    flags.pop(name, None)
    cfg = dict(maze=maze, total_steps=steps, num_envs=envs, num_steps=num_steps, minibatches=minibatches, seed=seed,
               ent_coef=ent_coef, history=history, lr=lr, lr_final=lr_final, eval_games=200, ckpt_every=10)
    print(f"starting '{name}' on {GPU}: {cfg}")
    print(train_remote.remote(name, cfg))


@app.local_entrypoint()
def stop(name: str):
    flags[name] = True
    print(f"asked '{name}' to stop; it will save, evaluate and exit within one update")
