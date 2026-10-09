"""PPO for the Pac-Man side (CleanRL style, one file). Writes metrics and checkpoints to --out.

    python -m ghostrl.train --maze small --total-steps 2_000_000 --out runs/pac_small
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from .env import GhostRLEnv
from .maze import load_maze
from .sim import Sim, SimConfig


@dataclass
class Config:
    maze: str = "arcade"
    num_envs: int = 64
    num_steps: int = 128
    total_steps: int = 2_000_000
    lr: float = 2.5e-4  # learning rate at the start; decays linearly to lr_final
    lr_final: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    epochs: int = 4
    minibatches: int = 4
    clip: float = 0.2
    ent_coef: float = 0.01
    history: bool = False  # trail planes + 2 extra scalars (see env.py)
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    reward_scale: float = 0.2  # rewards run to about +30 per clear; keeps the value targets modest
    max_ticks: int = 2000
    seed: int = 0
    device: str = "cpu"
    eval_games: int = 200
    ckpt_every: int = 25  # updates
    out: str = "runs/pacman"


class Net(nn.Module):
    """Small CNN over the grid planes + the scalar features -> masked policy and a value."""

    def __init__(self, H: int, W: int, n_planes: int = 10, n_scalars: int = 9, history: bool = False):
        super().__init__()
        self.history = history
        self.conv = nn.Sequential(
            nn.Conv2d(n_planes, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten())
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, n_planes, H, W)).shape[1]
        self.scal = nn.Sequential(nn.Linear(n_scalars, 32), nn.ReLU())
        self.trunk = nn.Sequential(nn.Linear(flat + 32, 128), nn.ReLU())
        self.pi, self.v = nn.Linear(128, 4), nn.Linear(128, 1)
        for layer in (self.pi,):
            nn.init.orthogonal_(layer.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)

    def forward(self, planes, scalars, mask):
        h = self.trunk(torch.cat([self.conv(planes), self.scal(scalars)], 1))
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)


def to_torch(obs, device):
    return (torch.as_tensor(obs["planes"], device=device).float(),
            torch.as_tensor(obs["scalars"], device=device),
            torch.as_tensor(obs["mask"], device=device))


@torch.no_grad()
def evaluate(net: Net, sim: Sim, games: int, device: str, greedy: bool = True, seed: int = 123) -> dict:
    """Play `games` full games against the classic ghosts; one game per environment."""
    env = GhostRLEnv(sim, games, history=net.history)
    obs = env.reset()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    stats = dict(ret=np.zeros(games), length=np.zeros(games), captured=np.zeros(games, bool),
                 cleared=np.zeros(games, bool), dots_left=np.zeros(games))
    finished = np.zeros(games, bool)
    while not finished.all():
        logits, _ = net(*to_torch(obs, device))
        a = logits.argmax(-1) if greedy else torch.multinomial(torch.softmax(logits, -1).cpu(), 1, generator=gen)[:, 0]
        obs, _, done, info = env.step(a.cpu().numpy())
        fin = info["final"]
        new = done & ~finished
        for k in stats:
            stats[k][new] = fin[k][new]
        finished |= done
    return dict(games=games, clear_rate=float(stats["cleared"].mean()), capture_rate=float(stats["captured"].mean()),
                mean_return=float(stats["ret"].mean()), mean_length=float(stats["length"].mean()),
                mean_dots_left=float(stats["dots_left"].mean()), greedy=greedy)


def train(cfg: Config, on_update=None) -> dict:
    """Run PPO. `on_update(update, out_dir)` is called after every update (the cloud job commits its volume
    there). Create an empty file named STOP in the output folder to end a run early; it still evaluates."""
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    from torch.utils.tensorboard import SummaryWriter

    tb = SummaryWriter(str(out / "tb"))
    tb.add_text("config", "```\n" + json.dumps(asdict(cfg), indent=2) + "\n```")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.device
    sim = Sim(load_maze(cfg.maze), SimConfig(max_ticks=cfg.max_ticks))
    m = sim.maze
    env = GhostRLEnv(sim, cfg.num_envs, history=cfg.history)
    net = Net(m.H, m.W, env.n_planes, env.n_scalars, cfg.history).to(dev)
    NP, NS = env.n_planes, env.n_scalars
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, eps=1e-5)

    T, N = cfg.num_steps, cfg.num_envs
    batch = T * N
    mb = batch // cfg.minibatches
    updates = max(cfg.total_steps // batch, 1)
    planes = torch.zeros((T, N, NP, m.H, m.W), dtype=torch.uint8, device=dev)  # counts are small ints
    scalars = torch.zeros((T, N, NS), device=dev)
    masks = torch.zeros((T, N, 4), dtype=torch.bool, device=dev)
    actions = torch.zeros((T, N), dtype=torch.long, device=dev)
    logp = torch.zeros((T, N), device=dev)
    rewards = torch.zeros((T, N), device=dev)
    dones = torch.zeros((T, N), device=dev)
    values = torch.zeros((T, N), device=dev)

    obs = env.reset()
    next_done = torch.zeros(N, device=dev)
    step_count, t0 = 0, time.time()
    recent = []  # (return, length, cleared, captured, dots_left) of finished episodes
    f = open(out / "metrics.csv", "w", newline="")
    w = csv.writer(f)
    w.writerow(["update", "steps", "sps", "episodes", "ep_return", "ep_length", "clear_rate", "capture_rate",
                "dots_left", "policy_loss", "value_loss", "entropy", "approx_kl", "explained_var"])
    last = {}

    for u in range(1, updates + 1):
        for g in opt.param_groups:
            g["lr"] = cfg.lr_final + (cfg.lr - cfg.lr_final) * (1 - (u - 1) / updates)
        for t in range(T):
            p, s, mk = to_torch(obs, dev)
            planes[t], scalars[t], masks[t], dones[t] = p.to(torch.uint8), s, mk, next_done
            with torch.no_grad():
                logits, v = net(p, s, mk)
                dist = Categorical(logits=logits)
                a = dist.sample()
            actions[t], logp[t], values[t] = a, dist.log_prob(a), v
            obs, r, done, info = env.step(a.cpu().numpy())
            rewards[t] = torch.as_tensor(r * cfg.reward_scale, dtype=torch.float32, device=dev)
            next_done = torch.as_tensor(done, dtype=torch.float32, device=dev)
            fin = info["final"]
            for i in np.flatnonzero(done):
                recent.append((fin["ret"][i], fin["length"][i], fin["cleared"][i], fin["captured"][i], fin["dots_left"][i]))
            step_count += N
        recent = recent[-500:]

        with torch.no_grad():  # GAE
            _, nv = net(*to_torch(obs, dev))
            adv = torch.zeros_like(rewards)
            lastgae = 0
            for t in reversed(range(T)):
                nonterm = 1.0 - (next_done if t == T - 1 else dones[t + 1])
                nxt = nv if t == T - 1 else values[t + 1]
                delta = rewards[t] + cfg.gamma * nxt * nonterm - values[t]
                adv[t] = lastgae = delta + cfg.gamma * cfg.gae_lambda * nonterm * lastgae
            ret = adv + values

        b_planes, b_scal, b_mask = planes.reshape(batch, NP, m.H, m.W), scalars.reshape(batch, -1), masks.reshape(batch, 4)
        b_act, b_logp, b_adv, b_ret, b_val = (x.reshape(batch) for x in (actions, logp, adv, ret, values))
        for _ in range(cfg.epochs):
            perm = torch.randperm(batch, device=dev)
            for i in range(0, batch, mb):
                idx = perm[i:i + mb]
                logits, v = net(b_planes[idx].float(), b_scal[idx], b_mask[idx])
                dist = Categorical(logits=logits)
                nlp = dist.log_prob(b_act[idx])
                ratio = (nlp - b_logp[idx]).exp()
                a_ = b_adv[idx]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                pl = torch.max(-a_ * ratio, -a_ * ratio.clamp(1 - cfg.clip, 1 + cfg.clip)).mean()
                vl = 0.5 * ((v - b_ret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                loss = pl - cfg.ent_coef * ent + cfg.vf_coef * vl
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), cfg.max_grad_norm)
                opt.step()
        with torch.no_grad():
            kl = ((ratio - 1) - (ratio.log())).mean().item()
            ev = 1 - ((b_ret - b_val).var() / (b_ret.var() + 1e-8)).item()
        sps = step_count / (time.time() - t0)
        R = np.array(recent, dtype=float).reshape(-1, 5)
        last = dict(update=u, steps=step_count, sps=sps, episodes=len(R),
                    ep_return=R[:, 0].mean() if len(R) else float("nan"),
                    ep_length=R[:, 1].mean() if len(R) else float("nan"),
                    clear_rate=R[:, 2].mean() if len(R) else float("nan"),
                    capture_rate=R[:, 3].mean() if len(R) else float("nan"),
                    dots_left=R[:, 4].mean() if len(R) else float("nan"),
                    policy_loss=pl.item(), value_loss=vl.item(), entropy=ent.item(), approx_kl=kl, explained_var=ev)
        w.writerow([f"{v:.5g}" if isinstance(v, float) else v for v in last.values()])
        f.flush()
        groups = dict(ep_return="episode/return", ep_length="episode/length", clear_rate="episode/clear_rate",
                      capture_rate="episode/capture_rate", dots_left="episode/dots_left", policy_loss="loss/policy",
                      value_loss="loss/value", entropy="loss/entropy", approx_kl="loss/approx_kl",
                      explained_var="loss/explained_variance", sps="speed/steps_per_second")
        for k, tag in groups.items():
            if not np.isnan(last[k]):
                tb.add_scalar(tag, last[k], step_count)
        tb.add_scalar("loss/learning_rate", opt.param_groups[0]["lr"], step_count)
        tb.flush()
        if on_update is not None:
            on_update(u, out)
        stop = (out / "STOP").exists()
        if u == 1 or u % 10 == 0 or u == updates:
            print(f"[{u}/{updates}] steps {step_count:,} sps {sps:,.0f} ret {last['ep_return']:.2f} "
                  f"len {last['ep_length']:.0f} clear {last['clear_rate']:.2f} dots_left {last['dots_left']:.0f} "
                  f"ent {last['entropy']:.2f} kl {kl:.4f}", flush=True)
        if u % cfg.ckpt_every == 0 or u == updates or stop:
            torch.save(dict(net=net.state_dict(), cfg=asdict(cfg), update=u), out / "checkpoint.pt")
        if stop:
            print(f"STOP file found: ending at update {u}", flush=True)
            break
    f.close()
    result = {"greedy": evaluate(net, sim, cfg.eval_games, dev, True), "sampled": evaluate(net, sim, cfg.eval_games, dev, False)}
    result["train_last"] = {k: (float(v) if not isinstance(v, int) else v) for k, v in last.items()}
    (out / "eval.json").write_text(json.dumps(result, indent=2))
    for mode in ("greedy", "sampled"):
        for k in ("clear_rate", "capture_rate", "mean_return", "mean_length", "mean_dots_left"):
            tb.add_scalar(f"eval_{mode}/{k}", result[mode][k], step_count)
    tb.close()
    if on_update is not None:
        on_update(-1, out)
    print("eval", json.dumps(result["greedy"]), flush=True)
    return result


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    for k, v in asdict(cfg).items():
        ap.add_argument("--" + k.replace("_", "-"), type=(lambda x: x.lower() in ("1", "true", "yes")) if isinstance(v, bool) else (lambda x: int(float(x))) if isinstance(v, int) else type(v), default=v)
    train(Config(**{k.replace("-", "_"): v for k, v in vars(ap.parse_args()).items()}))


if __name__ == "__main__":
    main()
