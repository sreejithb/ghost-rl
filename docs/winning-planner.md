# The winning Pac-Man planner: flat Monte-Carlo search

Source material for an HTML tutorial. Everything here describes code in this repository (`ghostrl/pacman.py`, class `SearchPlanner`, tuned with `ghostrl/tune.py`). Numbers are measured results; where something is a guess, it says so.

## 1. Is it Monte Carlo *tree* search? No.

It is **flat Monte-Carlo search**, also called rollout-based move evaluation:

- There is **no tree**. Nothing is remembered between the moves it considers, no node statistics, no UCB exploration formula, no tree growing deeper where the results look promising.
- For each of Pac-Man's (up to 4) possible next moves, it plays the game forward in a copy of the real simulator several times ("rollouts" or "futures"), scores how each future went, and averages. It then plays the move with the best average.
- Monte-Carlo **tree** search (MCTS, and its UCT variant) adds a growing tree on top of this idea. We built a UCT version too. It did *worse* here (0 mazes cleared in 100 games, against about 8% for the flat search), and was roughly 2 to 6 times slower per step, so it was dropped. See section 9.

A useful way to say it in the tutorial: *"Try each move, imagine many futures for each, pick the move whose futures went best on average. Re-do this from scratch every tick."*

## 2. The game, in the terms the tutorial needs

- **Maze:** the classic arcade layout, 28 columns × 31 rows, 300 walkable tiles, 244 dots (240 small dots plus 4 big power pellets).
- **Tick:** one time step. Pac-Man moves one tile per tick. Each ghost moves one tile per tick except that every ghost skips 1 tick in 16 (so ghosts are slightly slower).
- **Ghosts:** Blinky, Pinky, Inky and Clyde with the original 1980 targeting rules in chase mode. Scatter mode (ghosts head to their home corners) alternates with chase on a timer. Ghosts never reverse, except when the mode switches.
- **Power pellets:** eating one makes all ghosts frightened for 45 ticks: they wander randomly at half speed and Pac-Man can eat them.
- **Game ends** when Pac-Man is caught, eats all 244 dots (a "clear"), or hits the 2,000-tick limit.
- **Goal of the planner:** be a strong, *frozen* Pac-Man baseline for an experiment comparing two ghost teams (classic vs. learned). It must not secretly know the ghosts' rules.

## 3. How one decision works (the algorithm)

Run every tick, for every game being played, all games at once as one batch.

```
decide(state):
    for each legal first move m in {up, left, down, right}:      # at most 4
        repeat N times (N "futures"):
            s = copy of the real simulator state
            apply move m                                          # tick 1
            for tick 2 .. D (D = lookahead depth):
                Pac-Man picks a move with a cheap greedy rule     # section 4
                the ghosts move by the neutral model              # section 5
                step the REAL simulator forward one tick
                add up rewards                                    # section 6
            score = rewards - death_cost (if caught)
                    - dot_pull * (path distance to nearest dot at the end)
        Q[m] = average score over the N futures
    Q[m] += heading_bonus if m == Pac-Man's current direction
    choose argmax Q                                               # with probability 2%: a random legal move instead
```

Key facts:
- **Depth D and futures N** are the two main budget knobs. The winner uses D = 15 ticks and N = 16 (or N = 8 for the cheaper variant).
- Cost per tick per game is `4 × N × D` simulator steps: 960 for N=16, D=15; 480 for N=8, D=15. All games, moves and futures are stacked into one giant array and stepped together, which is why it is feasible in NumPy.
- **It reuses the real simulator.** Speeds, skip ticks, the scatter/chase clock (which pauses while frightened), power pellets, frightened mode and eaten-ghost respawns are therefore exact inside the futures, not approximated.
- **Common random numbers:** the 4 candidate moves of one game share the same random noise for future number 1, number 2, and so on. This makes the comparison between moves less noisy, because a move is not rewarded just for drawing luckier noise. (A standard variance-reduction trick.)
- **2% random moves** (`eps = 0.02`) are added so that repeated games against deterministic ghosts do not replay identically; this is what makes measuring a clear *rate* meaningful.

## 4. How Pac-Man moves *inside* a future (the rollout policy)

After the forced first move, Pac-Man needs a move every tick. Running a full planner inside every future would be far too slow, so a cheap greedy rule picks it. For each candidate next tile `n` (reversing is not allowed unless it is the only option):

```
score(n) =  fear_w   * min(path distance from n to the nearest live ghost, fear_cap)
          + dotnext_w * (1 if there is a dot on n else 0)
          - prey_w   * (path distance to the nearest frightened ghost, if any)
          + noise    * Uniform(0, 1)
```

then pick the tile with the highest score. In words: *step away from dangerous ghosts, step onto dots, step toward ghosts that are edible, and add some randomness so the many futures differ.*

- "Path distance" is the shortest-path length through the maze, from a precomputed 300 × 300 table.
- This rule is deliberately crude. It is not a good player on its own; it only needs to be good enough that the futures are realistic.
- Optional "pellet reserve" (`reserve`): subtract 8 for stepping on a power pellet when no live ghost is within 10 tiles and small dots remain. It is on in the 8-future variant and off in the 16-future winner.

## 5. How the ghosts are modelled inside a future

In chase mode, every ghost is replaced by a **neutral path-chaser**: it takes the legal move (no reversing) whose next tile is closest to Pac-Man **by path length**. Ties go to up, left, down, right.

This is the central design choice for the experiment:

- **The planner knows nothing about how any ghost team actually chooses moves.** It does not use Pinky's lookahead, Inky's reflection or Clyde's retreat. The same Pac-Man is therefore a fair opponent for both the classic team and a learned team.
- It is a pessimistic model ("every ghost simply hunts me"). Real classic ghosts also try to *ambush*, so the model is wrong in a specific way, but it favours neither team.
- Scatter mode and frightened mode are **scripted for both teams** in the simulator, so the futures run them exactly: the planner implicitly knows the scatter corners (public arcade rules) and, because the frightened walk is a deterministic function of the state, it can effectively predict the "random" frightened moves. This is a known small unfairness that applies equally to both teams.

## 6. What a future is worth (the scoring)

Over the D ticks of a future:

| Event | Score |
|---|---|
| Eat a dot (including a power pellet tile) | +1 |
| Eat a frightened ghost | + `ghost_bonus` (the winner uses **0**) |
| Clear the whole maze | +200 |
| Get caught | − `death_cost` (the winner uses 120) and the future stops |
| End of future | − `dot_pull` × path distance to the nearest remaining dot (0.05) |

The last term is a tie-breaker: if two moves are equally safe and eat the same dots, prefer the one that ends closer to food. Without it, Pac-Man may idle in safe areas.

Then, per first move: average over the N futures, add `heading_bonus` if the move continues his current direction (the winner uses 0, the cheaper variant 1), and pick the best.

## 7. The tuned settings (the winners)

Found by random search (section 8). All settings tested against the classic ghosts, 1,000 fresh games each.

| | **Original** | **Winner A (16 futures)** | **Winner B (8 futures, cheaper)** |
|---|---|---|---|
| Futures per move N | 8 | **16** | 8 |
| Lookahead depth D (ticks) | 25 | **15** | **15** |
| death_cost | 60 | **120** | 240 |
| ghost_bonus | 6 | **0** | **0** |
| dot_pull | 0.3 | **0.05** | 0.15 |
| fear_w / fear_cap | 3 / 6 | **3 / 6** | 1.5 / 10 |
| dotnext_w | 2 | **2** | 4 |
| noise | 1.0 | **0.5** | 2.0 |
| heading_bonus | 0.3 | **0** | 1 |
| pellet reserve | off | **off** | on |
| random-move rate | 2% | 2% | 2% |
| prey_w (not tuned) | 0.5 | 0.5 | 0.5 |
| **Mazes cleared (1,000 games)** | 79 (7.9%) | **180 (18.0%)** | **190 (19.0%)** |
| Dots left on average (of 244) | 45.4 | **37.1** | 39.9 |
| Ticks per game | 371 | 344 | 402 |
| Ghosts eaten per game | 5.5 | 2.7 | 2.2 |
| Speed in a 2-CPU Modal container | | 100 steps/s | 188 steps/s |

(Winner A is the planned frozen setting; Winner B is the planned held-out variant. Neither is committed to a config file yet.)

**Generalisation check** (300 games each, ghosts the planner was *not* tuned on):

| Ghosts | Original | Winner B | **Winner A** |
|---|---|---|---|
| Path-chasing | 92/300 (30.7%) | 134/300 (44.7%) | **159/300 (53.0%)** |
| Random-move | 224/300 (74.7%) | 256/300 (85.3%) | **271/300 (90.3%)** |

So the improvement is not specific to the classic ghosts. Clear rate by opponent: random ghosts are easy, path-chasers medium, classic ghosts hardest. The classic ghosts' *ambush* targeting is what beats the planner most often.

## 8. How the settings were found

- **Search space:** N ∈ {4, 8, 16}; D ∈ {15, 25, 40}; death_cost ∈ {30, 60, 120, 240}; ghost_bonus ∈ {0, 3, 6, 12}; dot_pull ∈ {0, 0.05, 0.15, 0.3, 0.6}; fear_w ∈ {1.5, 3, 6}; fear_cap ∈ {4, 6, 10}; dotnext_w ∈ {1, 2, 4}; noise ∈ {0.5, 1, 2}; heading_bonus ∈ {0, 0.3, 1}; reserve ∈ {off, on}.
- **Stage 1:** the original setting plus 24 random draws, 400 games each, run in parallel on Modal CPU containers.
- **Stage 2:** the original plus the top 3 by clear rate plus the top by return, 1,000 fresh games each (different random seeds).
- **Stage 3:** the best two against path-chasing and random-move ghosts.
- **A lesson worth a tutorial callout:** ranking by *return* (a reward that counts +1 per ghost eaten) picked the wrong winner. The top return (23.0) belonged to a setting that cleared only about 10%. Ranking by *clear rate* picked settings with a return of about 21 but twice the clears. A metric must match the goal.
- **Patterns in the winners** (observations, not proven causes): all top settings had a short 15-tick lookahead and `ghost_bonus = 0` (do not hunt ghosts). A plausible explanation is that chasing ghosts pulls Pac-Man into danger, and that long rollouts with a crude greedy policy accumulate error. Neither idea was tested directly.

## 9. Related planners built and compared

| Planner | Idea | Mazes cleared vs. classic ghosts | Dots left | Verdict |
|---|---|---|---|---|
| **Flat Monte-Carlo, tuned (winner A)** | Section 3 | **18.0% (1,000 games)** | 37 | Best |
| Flat Monte-Carlo, original settings | same, hand-picked numbers | 4.5% to 7.9% | 45 to 49 | Baseline |
| UCT tree search (32 or 64 iterations) | A real tree with UCB, greedy rollouts at leaves | 0% (100 games each) | 50 to 57 | Worse and roughly 2 to 6 times slower |
| Territory planner | For each move, count tiles Pac-Man reaches before any ghost could; stay in large safe areas; head for the nearest dot | 0% (200 games) | 83 | Lives long (702 ticks) but eats little |

**Territory planner in one paragraph.** For each ghost it computes the earliest tick it could reach every tile (shortest path with no reversing, converted to ticks using the known 1-in-16 skip), takes the minimum over ghosts, and then finds the tiles Pac-Man could reach strictly sooner along paths that stay safe. It picks the move that keeps the biggest safe area, then heads for the nearest dot. It assumes nothing about ghost decision-making, so it is "safe" against any ghost type, but it cannot see two ghosts closing a pincer and it tends to circle: against random ghosts 93 of 100 games hit the 2,000-tick limit.

**An earlier diagnosis of how the planners die** (made before tuning): trouble builds over a few ticks, not suddenly. Typically a ghost sits 2 to 4 tiles behind Pac-Man for 10 or more ticks, a second ghost arrives from the other side, and the safe area shrinks to nothing. Some games also show "dithering": Pac-Man oscillating between two tiles when all moves look equally bad.

## 10. Reinforcement learning, for context

The same Pac-Man was also trained with PPO (about 40 runs). The best RL runs left about 50 dots on average and cleared **0 of 6,200+ evaluation games**; the tuned planner clears 18%. This is why the planner, not an RL agent, is the frozen baseline. (A separate document, `docs/project-summary.md`, covers the whole project.)

## 11. Known weaknesses and honest caveats

- **18% is below the project's original 30 to 60% target** against the classic ghosts.
- **Model mismatch:** the futures assume path-chasing ghosts; the real classic ghosts ambush. Moves that look safe under the model can be unsafe in the real game.
- **Tuned only against the classic ghosts**, with a partial generalisation check (two other ghost types). The planned held-out variant is meant to catch overfitting.
- **Crude rollout policy:** the greedy rule inside futures is simple; a smarter one is the most likely source of further gains.
- **No memory between ticks:** the whole search is redone every tick, with no reuse.
- **Slow:** about 100 to 190 simulator steps per second per 2-CPU cloud container, so a 1,000-game evaluation takes 35 to 57 minutes there. (On the author's laptop the original setting ran about 460 steps/s.)
- **Foresight of scripted rules:** see section 5.
- **Statistics:** 400-game stage-1 estimates are ±2 percentage points; 1,000-game confirmations about ±1.2 points. Differences smaller than that are noise.

## 12. Where the code is

- `ghostrl/pacman.py`: `SearchPlanner` (this planner), `Planner` (territory planner), `UCTPlanner`.
- `ghostrl/tune.py`: the search space, `sample_configs`, `run_config` (plays N games and reports clears, dots left, ticks, return, ghosts eaten).
- `ghostrl/tune_launch.py` and `ghostrl/modal_app.py`: launch and collect the cloud tuning jobs.
- `docs/planner_results.html`: a table of every planner and RL run.
- Pictures that could be reused: `docs/img/graph_state.png`, `docs/img/graph_spread.png`, `docs/img/option1_state.png` (the maze as a graph and per-move features), and GIFs of games in the repo root (`pacman_best.gif` etc.; GIFs are not committed).
- At the time of writing, the tuned settings are not yet written to `configs/pacman_frozen.yaml`, and the tuning code is not yet committed.

Reproduce a measurement (small example, from the repo root):

```python
from ghostrl.tune import run_config
winner_a = dict(n_rollouts=16, depth=15, death_cost=120.0, ghost_bonus=0.0, dot_pull=0.05,
                fear_w=3.0, fear_cap=6.0, dotnext_w=2.0, noise=0.5, heading_bonus=0.0, reserve=False)
print(run_config(winner_a, games=200, seed=1))   # slow on a laptop; use Modal for 1,000 games
```

## 13. Glossary for the tutorial

- **Rollout / future:** one imagined continuation of the game from the current state.
- **Horizon / depth:** how many ticks a future runs.
- **Flat Monte-Carlo vs. Monte-Carlo tree search:** both average random-ish futures; the tree version also grows a tree of explored moves and spends more effort on promising branches.
- **Rollout policy:** the cheap rule that picks Pac-Man's moves inside a future.
- **Common random numbers:** sharing the same randomness across the alternatives being compared, so differences reflect the alternatives, not the luck.
- **Path distance:** shortest walking distance in the maze, not straight-line distance.
- **Clear:** eating all 244 dots; **caught:** touching a live ghost; **tick:** one game step.
- **Neutral model:** assuming ghosts simply chase by shortest path, so the planner favours no ghost team.
- **Frozen / held-out:** a frozen Pac-Man never changes once ghost training starts; a held-out variant is used only for testing.

## 14. Suggested outline for the HTML tutorial

1. The problem: a Pac-Man that does not know how the ghosts think (and why).
2. Flat Monte-Carlo in one picture: 4 moves × several futures each, average, pick the best.
3. Walk through one decision with a small made-up example (illustrative numbers: for the move "left", 4 futures score 12, 9, −110, 14, so the average is −18.8; for "right" 15, 11, 13, 10, so 12.3; pick "right").
4. Inside a future: the greedy rule for Pac-Man, the path-chaser ghosts, the real simulator.
5. The scoring table and why `ghost_bonus = 0` won.
6. Tuning: random search, why clear rate beat return as the metric.
7. Results: original vs. winner, three ghost types, a bar chart.
8. What it cannot do, and why tree search did not help.
9. Interactive idea: sliders for depth and number of futures that show the cost `4 × N × D` and the measured clear rate for the tested settings.
