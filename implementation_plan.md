# Deep RL Assignment 1 — Complete Implementation Plan

## Overview

Build two fully self-contained, richly-commented Jupyter notebooks that satisfy every graded requirement for Group 91. The existing Python modules in `mab/` and `dp/` are a useful starting point but are scaffolds only — they need significant extensions. Each notebook must be **completely executable top-to-bottom**, produce all required outputs, and contain the timestamp / VM-ID header cell.

---

## Key Parameters (Group 91)

| Parameter | Value |
|---|---|
| `GROUP_NUMBER` | 91 |
| `LAST_DIGIT` | 1 |
| `MAB_NUM_MEDICINES (K)` | `(91 % 3) + 5 = 6` |
| `MAB_HIDDEN_PROBS` | `[0.4 + ((91+i) % 6)*0.07 for i in range(6)]` |
| `MAB_NUM_PATIENTS` | 1000 |
| `DP_GRID_SIZE` | 5 (last digit ≤ 4) |
| `DP_MAX_BATTERY` | 15 (last digit is odd) |
| `DP_WIND_PROB` | 0.20 (last digit ≤ 4) |
| `DP_MAX_STEPS` | 50 (5×5 grid) |
| `DP_NUM_RESCUE` | 2 |
| `DP_NUM_CHARGERS` | 1 |
| `DP_NUM_DANGER` | 3 |
| `DP_NUM_BLOCKED` | 2 |

---

## Part 1 — MAB Solution Notebook (`MAB_Solution.ipynb`)

### What needs to change / be added

#### Cell 0 — Header cell (REQUIRED by assignment)
- Print execution timestamp, Virtual Machine ID, Group Number (already exists — verify it's first).

#### Task 1 — Dataset Design (1 mark)
Current `dataset.py` creates the scaffold; need to ensure:
- `random.seed(G)` AND `numpy.random.seed(G)` are both called.
- Display group number, K medicines, and hidden success probabilities clearly.
- Print first 10 rows of the dataset.
- Add rich print formatting so it looks presentable.

#### Task 2 — Immediate Exploitation / Greedy Strategy (1 mark)
Current `greedy_strategy()` in `strategies.py` works but needs:
- Better inline comments explaining the 10-pull exploration phase.
- A per-step printout showing which arm was selected and cumulative reward progression.
- Summary output: final cumulative reward, best arm identified.

#### Task 3 — Epsilon-Greedy Strategy (1.5 marks)
Current `epsilon_greedy_strategy()` exists but needs:
- Run for ε ∈ {0.01, 0.10, 0.50} separately.
- Per-epsilon analysis output: final reward, number of times optimal arm was selected.
- Comparison sub-plot showing how epsilon affects performance.

#### Task 4 — UCB1 Strategy (1 mark)
Current `ucb1_strategy()` exists but needs:
- Inline explanation of the UCB1 formula inside the code comments.
- Summary output: which arm UCB1 converges to, confidence bound values.

#### Task 5 — Comparative Analysis (0.5 marks)
Current `plot_cumulative_rewards()` is basic; needs:
- Proper styled matplotlib figure (grid, legend, title, axis labels, colors).
- Printed answers to all 4 comparative questions.
- 3–5 sentence written summary embedded as a markdown cell.

#### Supporting module changes needed in `mab/`:
- **`dataset.py`**: Add `random.seed(G)` call alongside `numpy.random.seed(G)`.
- **`analysis.py`**: Enhance `plot_cumulative_rewards` with colors, markers, annotations.
- **`strategies.py`**: Add richer return values (per-step arm selection tracking).

---

## Part 2 — DP Solution Notebook (`DP_Solution.ipynb`)

### What needs to change / be added

#### Cell 0 — Header cell (REQUIRED)
- Print timestamp, VM ID, Group ID (already exists).

#### Section 1 — Custom Drone Rescue Environment (1 mark)
Current `env.py` has `_step_deterministic`, `_apply_wind`, `enumerate_reachable_states` but is **missing** the required interface:
- **`reset()`** — must return initial state.
- **`step(action)`** — public interface wrapping `_step_deterministic` / `_apply_wind`.
- **`render()`** — must print/display the current grid state with drone position.
- **The wind logic in `_apply_wind` is broken** — it returns a single state instead of a proper stochastic distribution. For DP, we need `get_transitions(state, action)` returning `[(prob, next_state, reward)]`.
- **Charging station logic**: entering C sets battery to full (correct); hovering on C adds +2 (correct per spec).
- Add `render()` that prints the grid with current drone position highlighted.

#### Section 2 — Dynamic Programming / Value Iteration (2 marks)
Current `mdp.py` has basic value iteration but needs:
- Fix `enumerate_reachable_states()` — currently only uses `_step_deterministic`, not wind transitions.
- Fix value iteration to handle **stochastic transitions** (wind zones) — must use `get_transitions()` instead of calling `_step_deterministic` directly.
- Print convergence iterations, runtime, and final delta.
- Add a convergence plot (delta vs iteration).

#### Section 3 — Policy Visualisation (1 mark)
Current `visualization.py` has basic arrow/heatmap but needs:
- Color-coded grid background (danger=red, charging=blue, rescue=green, wind=yellow, blocked=grey).
- Arrow overlay showing policy directions.
- A step-by-step trajectory rollout printout showing the greedy path.

#### Section 4 — State-Value Analysis (1 mark)
Current `plot_value_heatmap()` exists but needs:
- Multiple heatmaps: vary battery level (e.g., battery=5, 10, 15) with fixed rescue mask.
- Written explanation of patterns observed.

#### Section 5 — DP Scalability Discussion (1 mark)
- New markdown cell with thorough structured discussion of:
  - Curse of Dimensionality formula.
  - State space size for 10×10 grid.
  - Why DP fails at scale.
  - How Deep RL (DQN, PPO) helps.
  - Real-world drone system implications.

#### Supporting module changes needed in `dp/`:
- **`env.py`**: Add `reset()`, `step(action)`, `render()`, fix `_apply_wind()` to return proper stochastic transitions via `get_transitions(state, action)`.
- **`mdp.py`**: Update `value_iteration()` to use `get_transitions` for proper stochastic Bellman updates; track convergence delta history.
- **`visualization.py`**: Enhance `plot_policy()` with color backgrounds and `plot_value_heatmap()` with numeric annotations; add `plot_trajectory()`.
- **`analysis.py`**: Add `plot_convergence(delta_history)` for convergence curve.

---

## Proposed Changes — File by File

### MAB Module (`mab/`)

#### [MODIFY] `mab/dataset.py`
- Add `import random`
- Add `random.seed(group_number)` alongside `np.random.seed`

#### [MODIFY] `mab/strategies.py`
- Add per-step arm tracking to all strategies
- Add richer comments explaining each algorithm

#### [MODIFY] `mab/analysis.py`
- Enhance `plot_cumulative_rewards()` with styled figure (colors, markers, grid, annotations)
- Add `print_comparison_summary()` function

#### [MODIFY] `MAB_Solution.ipynb`
- Complete rewrite: all 5 tasks with rich outputs, markdown explanations, and comparative analysis answers

---

### DP Module (`dp/`)

#### [MODIFY] `dp/env.py`
- Add public `reset()`, `step(action)` (wraps deterministic/stochastic), `render()`
- Add `get_transitions(state, action)` returning `[(prob, next_state, reward, done)]`
- Fix wind stochastic logic

#### [MODIFY] `dp/mdp.py`
- Update `value_iteration()` to use `get_transitions` for proper stochastic Bellman updates
- Track convergence history (delta per iteration)

#### [MODIFY] `dp/visualization.py`
- Add color-coded background to `plot_policy()`
- Add `plot_trajectory()` for rollout visualization
- Enhance `plot_value_heatmap()` with numeric annotations

#### [MODIFY] `dp/analysis.py`
- Add `plot_convergence(delta_history)` for convergence curve

#### [MODIFY] `DP_Solution.ipynb`
- Complete rewrite: all 5 sections with environment demo, DP results, visualizations, state-value analysis, and scalability discussion

---

## Verification Plan

### Automated
- Run `jupyter nbconvert --to notebook --execute MAB_Solution.ipynb` and `DP_Solution.ipynb` to verify they execute without errors.

### Manual Verification Checklist
- [ ] Task 1: Dataset printed with 10 rows, group params shown
- [ ] Task 2: Greedy strategy runs 1000 iterations, cumulative reward printed
- [ ] Task 3: ε = 0.01, 0.10, 0.50 all run; comparison shown
- [ ] Task 4: UCB1 runs, converges to best arm
- [ ] Task 5: Comparative plot and 4 questions answered
- [ ] DP Section 1: `reset()`, `step()`, `render()` all callable and working
- [ ] DP Section 2: Value iteration converges, prints iterations/runtime/delta
- [ ] DP Section 3: Policy visualized on colored grid
- [ ] DP Section 4: Value heatmap for multiple battery levels
- [ ] DP Section 5: Scalability discussion written
