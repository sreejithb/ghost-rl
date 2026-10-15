# Ghost RL: project summary (status as of 2026-10-13)

Personal research project by Sreejith B. Prepared as source material for resume bullets. Every number below was measured in this project; the caveats section says what must not be overstated.

## The question

Do Pac-Man ghosts trained with multi-agent reinforcement learning end up behaving like the four original 1980 ghosts (Blinky, Pinky, Inky, Clyde)?

The plan compares two ghost teams against one frozen Pac-Man agent:
- **Classic team:** the original targeting rules (Blinky chases directly, Pinky aims 4 tiles ahead with the arcade's up-facing bug, Inky reflects Blinky's position through a pivot, Clyde retreats when close).
- **Learned team:** four ghosts trained with multi-agent PPO (centralised training, decentralised execution).

Both teams play under identical rules, so the only difference is who picks the ghosts' moves in chase mode. The final evaluation measures outcomes (capture rate, time to capture), how closely the learned ghosts' choices match the classic rules, and team-level formation behaviour.

## What has been built (done)

**1. A fast, tested Pac-Man simulator (Python, NumPy).**
- Batched tile-grid simulator: thousands of games advance at once with array operations and no Python loops over games.
- Full 28 × 31 arcade maze (244 dots), classic ghost rules, scripted scatter/chase clock with arcade level-1 timings, ghost speed at 15/16 of Pac-Man's, power pellets and frightened mode.
- Frightened mode and the scatter/chase schedule are scripted identically for both teams, so the learned policy only controls chase mode.
- 71 automated tests, including a "golden replay" tripwire: a recorded 999-tick game that must replay identically, covering all seven mode switches, power pellets and eaten ghosts. I found and fixed a weak original snapshot (29 ticks, all scatter) that would not have caught a broken chase rule.
- GIF renderer that turns a seed and action log into an animated replay.

**2. Pac-Man baseline players, two approaches.**
- *Planners (no training).* A safe-area planner that computes, per tile, whether Pac-Man can reach it before any ghost could. A flat Monte-Carlo search that plays short futures in the real simulator and picks the best first move. A UCT tree search (worse than the flat search, dropped). The planners are written to know nothing about how ghosts choose moves; they assume neutral path-chasing ghosts.
- *Reinforcement learning.* A single-file masked PPO in PyTorch with a small CNN, TensorBoard logging, checkpointing, early-stop and resume.

**3. Planner tuning on cloud compute.** A random search over 25 settings (400 games each) on Modal CPU containers, followed by a 1,000-game confirmation of the finalists and a test against ghost types the planner was not tuned on.

| Ghosts | Original planner | Tuned planner |
|---|---|---|
| Classic (1,000 games) | 7.9% of mazes cleared | 18.0% to 19.0% cleared |
| Path-chasing (300 games) | 30.7% | 53.0% |
| Random-move (300 games) | 74.7% | 90.3% |

The gain held on ghost types it was not tuned on. The winning settings use a short 15-tick lookahead and no reward for eating ghosts.

**4. RL training infrastructure on Modal (cloud).**
- A reinforcement-learning environment wrapper that supports either side being learned (Pac-Man or ghosts) over the same simulator step, with per-side rewards, action masks and auto-reset. About 200,000 environment steps per second on one CPU core.
- Training on a Modal T4 GPU at roughly 8,000 to 11,000 steps per second, against 300 to 500 on an 8-core laptop CPU (about 20 times faster).
- Live TensorBoard monitoring with log syncing from a Modal volume, and a stop-file mechanism to end a run early.
- Diagnosed why long Modal jobs were being cancelled when the local connection dropped, and moved long jobs to a deployed app with spawned calls that survive client disconnects.

**5. RL experiments (about 40 training runs).** Studied learning rate and schedule, entropy bonus (0.01 to 0.25), discount (0.9, 0.99, 0.999), batch size, reward design (original small values, arcade point values, no end-of-game reward with a per-tick cost), trail planes as extra state (A/B test, 2 to 3 seeds each) and continued training to 10M steps.
- A higher learning rate with larger batches helped most.
- Discount 0.9 and entropy 0.25 were clearly worse; trail planes gave no reliable gain.
- During this work I caught a measurement flaw in my own evaluation: "greedy" evaluation against deterministic ghosts is one repeated game, not 200. I re-reported all results using sampled-move evaluation and flagged an earlier "best" figure that a third seed had shown to be optimistic.

## Where things stand

- The best RL Pac-Man leaves about 50 of 244 dots on average and has not cleared a full maze in any of 6,200+ evaluation games. It roughly matches the early planner on dots left, and is behind the tuned planner.
- The tuned planner is the strongest baseline and is the candidate for the frozen Pac-Man.
- No ghost training has been done yet. The ghost-learning half of the project is the main pending piece.

## Pending

1. **Freeze Pac-Man.** Write the winning planner settings to `configs/pacman_frozen.yaml` (committed, never changed after ghost training starts) and choose a held-out second variant for testing only. Candidate: 16 futures × 15 ticks lookahead, death cost 120, ghost bonus 0, dot pull 0.05, 2% random moves.
2. **Ghost training.** Independent PPO as the baseline, MAPPO (shared critic, decentralised actors) as the main method, and a single controller for all four ghosts as the upper bound. Arcade maze, 5 conditions × 5 seeds on Modal.
3. **Evaluation.** Outcome metrics against both Pac-Man variants (the frozen one and the held-out one), how well learned ghost moves match the classic targeting rules, and team shape (do they spread and pincer like the originals).
4. **Write-up.** Side-by-side GIFs of classic and learned teams and a rule-match chart.
5. **Optional improvements considered:** an AlphaZero-style search with a learned value to strengthen Pac-Man, and a graph neural network input for the maze.

## Caveats (please do not overstate)

- The original goal for the Pac-Man baseline was a 30 to 60% clear rate against the classic ghosts. The tuned planner reaches 18 to 19% there (and 53% against simpler ghosts). That target is not met against the classic ghosts.
- RL Pac-Man did not clear any maze. Its reward values are my own untuned choices, not taken from a published system.
- The central research question, whether learned ghosts match the classic ones, is not answered yet. No learned ghosts exist yet.
- RL results use 2 to 3 seeds per setting, and seeds in one setting often differ by more than the gaps between settings.
- The planner results are measured only against ghosts in my own simulator, which is tile-based and simplified. It leaves out the ghost house, Cruise Elroy, fruit and extra lives.
- Planner settings were tuned on the classic ghosts; the other ghost types are a partial check of generalisation.

## Tech stack

Python 3.12, NumPy (simulator, planners), PyTorch (PPO), pytest (71 tests), TensorBoard, Modal (cloud CPU and GPU jobs, volumes, deployed apps), Pillow (GIF rendering), Git/GitHub.

## Resume-ready facts (all verifiable above)

- Built a batched NumPy Pac-Man simulator with the original ghost AI, power pellets and a scripted mode clock; 71 tests including a golden-replay regression test.
- Wrote a single-file masked PPO in PyTorch and ran about 40 experiments on Modal GPUs (roughly 20 times faster than laptop CPU), with live TensorBoard monitoring.
- Tuned a Monte-Carlo-search Pac-Man planner with parallel cloud search (25 settings, 1,000-game confirmation): maze-clear rate rose from 7.9% to about 19% against the classic ghosts, and from 31% to 53% and from 75% to 90% against two ghost types it was not tuned on.
- Found and corrected a flaw in my own evaluation protocol (a deterministic "greedy" evaluation that was really one repeated game), then re-reported all results with sampled evaluation.
- In progress: training a four-agent ghost team with MAPPO to test whether it rediscovers the classic ghosts' behaviour.
