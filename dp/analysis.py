"""
Analysis Helpers for DP Drone Rescue Assignment (Group 91).

This module provides functions for summarising and plotting DP results:
  - summarize_dp_results   : Print formatted convergence and runtime summary.
  - plot_convergence       : Plot delta (max value change) vs iteration number.
  - print_scalability_discussion : Print the curse of dimensionality analysis.
"""

from typing import Dict, List
import numpy as np
import matplotlib.pyplot as plt


def summarize_dp_results(metadata: Dict) -> str:
    """Build and return a formatted summary of Dynamic Programming results.

    Formats key convergence metrics into a readable string block for display
    in the Jupyter notebook output.

    Args:
        metadata (Dict): Results dictionary with keys:
            'iterations' : Number of VI iterations until convergence.
            'runtime'    : Wall-clock time in seconds.
            'delta'      : Final max |V_new - V_old| at convergence.
            'start_value': V*(start state) — expected return from the start.
            'n_states'   : Total number of reachable states explored.

    Returns:
        str: Formatted multi-line summary string.
    """
    sep = "=" * 60
    lines = [
        sep,
        "  Dynamic Programming — Value Iteration Results (Group 91)",
        sep,
        f"  Convergence iterations  : {metadata.get('iterations', 'N/A')}",
        f"  Runtime                 : {metadata.get('runtime', 0.0):.4f} seconds",
        f"  Final delta (max ΔV)    : {metadata.get('delta', 0.0):.2e}",
        f"  Convergence threshold θ : 1e-3",
        f"  Discount factor γ       : 1.0 (undiscounted)",
        f"  Reachable states        : {metadata.get('n_states', 'N/A'):,}",
        f"  V*(start state)         : {metadata.get('start_value', 0.0):.4f}",
        sep,
    ]
    return "\n".join(lines)


def plot_convergence(
    delta_history: List[float],
    theta: float = 1e-3,
    title: str = "Value Iteration Convergence — Group 91",
):
    """Plot the maximum value change (delta) per iteration of Value Iteration.

    Visualises how quickly the algorithm converges by plotting max|ΔV| on a
    logarithmic y-axis. A horizontal dashed line marks the convergence threshold θ.

    Args:
        delta_history (List[float]): Max |V_new - V_old| at each iteration.
                                      Produced by value_iteration().
        theta (float): Convergence threshold to mark on the plot.
        title (str): Plot title.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot delta on log scale (reveals convergence speed clearly)
    ax.semilogy(
        range(1, len(delta_history) + 1),
        delta_history,
        color="#2980B9",
        linewidth=2.2,
        label="Max |ΔV| per iteration"
    )

    # Mark the convergence threshold
    ax.axhline(
        y=theta, color="#E74C3C", linestyle="--", linewidth=1.5,
        label=f"Convergence threshold θ = {theta:.0e}"
    )

    # Mark the iteration where convergence was achieved
    if delta_history and delta_history[-1] < theta:
        convergence_iter = len(delta_history)
        ax.axvline(
            x=convergence_iter, color="#27AE60", linestyle=":", linewidth=1.5,
            label=f"Converged at iteration {convergence_iter}"
        )
        ax.annotate(
            f"Converged\n(iter {convergence_iter})",
            xy=(convergence_iter, delta_history[-1]),
            xytext=(convergence_iter + max(1, len(delta_history) // 20), theta * 10),
            fontsize=9, color="#27AE60",
            arrowprops=dict(arrowstyle="->", color="#27AE60"),
        )

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Max |ΔV| (log scale)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def print_scalability_discussion():
    """Print a structured discussion on DP scalability and the Curse of Dimensionality.

    Covers:
      1. State space formula and current size for Group 91.
      2. How state space grows for a 10×10 grid.
      3. Impact of more rescue targets and dynamic weather.
      4. Why DP becomes intractable at scale.
      5. How Deep RL methods (DQN, PPO) address these limitations.
      6. Connection to real-world autonomous drone systems.
    """
    print("=" * 70)
    print("  DP SCALABILITY DISCUSSION — Curse of Dimensionality (Group 91)")
    print("=" * 70)

    print("""
──────────────────────────────────────────────────────────────────────
1. CURRENT STATE SPACE (Group 91, 5×5 Grid)
──────────────────────────────────────────────────────────────────────

State representation: (row, col, battery, rescue_mask, step)

  Component          | Values        | Formula
  ─────────────────────────────────────────────
  Grid positions     | 25            | 5 × 5
  Battery levels     | 16            | 0 to 15 (inclusive)
  Rescue bitmask     | 4             | 2^2 = 4 (2 targets)
  Step count         | 51            | 0 to 50 (inclusive)
  ─────────────────────────────────────────────
  Upper bound        | 82,400 states | 25 × 16 × 4 × 51

  Many of these states are unreachable (e.g., drone can't be at step 50
  with full battery). BFS enumeration finds only the reachable subset.

──────────────────────────────────────────────────────────────────────
2. SCALING TO A 10×10 GRID
──────────────────────────────────────────────────────────────────────

If the grid grows to 10×10 with proportionally more objects:
  Component          | Current (5×5) | 10×10 Grid
  ─────────────────────────────────────────────
  Grid positions     | 25            | 100
  Battery levels     | 16            | ~31 (scaled)
  Rescue targets     | 2 → mask=4    | 6 → mask=64
  Step count         | 51            | ~201
  ─────────────────────────────────────────────
  Upper bound        | ~82,400       | ~400,000,000+

  This represents a ~5,000× increase in state space — clearly intractable
  for tabular methods that store a value for every state.

──────────────────────────────────────────────────────────────────────
3. MORE RESCUE TARGETS
──────────────────────────────────────────────────────────────────────

The rescue bitmask grows EXPONENTIALLY with the number of targets:
  - 2 targets  → 2² =    4 masks
  - 5 targets  → 2⁵ =   32 masks
  - 10 targets → 2¹⁰ = 1,024 masks
  - 20 targets → 2²⁰ = 1,048,576 masks

Even with a fixed grid and battery, 20 rescue targets alone creates
over 1 million sub-states — all combinations of "rescued / not rescued".

──────────────────────────────────────────────────────────────────────
4. DYNAMIC WEATHER CONDITIONS
──────────────────────────────────────────────────────────────────────

If weather becomes dynamic (e.g., wind zones shift over time, storm
cells appear/disappear), the environment is no longer stationary.
The MDP transition function P(s'|s,a) itself changes, making pre-computed
DP solutions invalid. The state space must then include weather state,
further multiplying the complexity.

──────────────────────────────────────────────────────────────────────
5. WHY DP BECOMES DIFFICULT (Curse of Dimensionality)
──────────────────────────────────────────────────────────────────────

Dynamic Programming suffers from:

  a) MEMORY: Storing V*(s) for millions of states requires gigabytes of RAM.
  b) COMPUTATION: Each VI iteration visits every state × every action —
     O(|S| × |A|) operations per iteration. With |S| = 10⁸, this is
     prohibitive even for modern hardware.
  c) ENUMERATION: BFS state enumeration itself becomes infeasible when
     the state space is too large to fit in memory.
  d) STATIONARITY ASSUMPTION: DP requires a fixed P(s'|s,a) — violated
     in real-world dynamic environments.

──────────────────────────────────────────────────────────────────────
6. HOW DEEP RL METHODS HELP
──────────────────────────────────────────────────────────────────────

Deep Reinforcement Learning replaces the state-indexed value table with
a neural network approximation V_θ(s) ≈ V*(s):

  - DQN (Deep Q-Network): Uses a CNN/MLP to approximate Q*(s,a) directly.
    Samples experience replay to break correlations. Scales to large
    state spaces like Atari games with pixel observations.

  - PPO (Proximal Policy Optimization): Directly learns a policy network
    π_θ(a|s) using policy gradients. Suitable for continuous action spaces
    and complex observations (e.g., camera feeds from a real drone).

  - Advantage:  No explicit state enumeration needed — the network
    generalises across similar states. Can handle dynamic environments
    by continual online learning.

  - Disadvantage: Less interpretable, harder to certify for safety-critical
    applications (e.g., hospital/rescue drones in regulated airspace).

──────────────────────────────────────────────────────────────────────
7. REAL-WORLD AUTONOMOUS DRONE SYSTEMS
──────────────────────────────────────────────────────────────────────

Real rescue drones face challenges that go far beyond this simulation:
  - Continuous 3D state space (position, velocity, orientation).
  - Battery dynamics depend on payload, wind speed, temperature.
  - Partial observability (camera view, sensor noise).
  - Multi-drone coordination (multi-agent MDP).
  - Regulatory constraints (airspace, no-fly zones).

In practice, hybrid approaches are used:
  - DP/MPC (Model Predictive Control) for short-horizon planning where
    the model is known.
  - Deep RL for long-horizon policy learning from simulation.
  - Transfer learning to bridge sim-to-real gaps.

For our 5×5 grid problem, tabular DP (Value Iteration) is well-suited.
But to deploy at real-world scale, Deep RL with function approximation
is the necessary next step.
""")
    print("=" * 70)
