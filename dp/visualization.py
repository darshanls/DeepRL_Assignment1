"""
Visualization Utilities for DP Drone Rescue Assignment (Group 91).

This module provides richly-styled plotting functions for:
  1. plot_grid_layout     : Colored grid showing all cell types.
  2. plot_policy          : Policy arrows overlaid on the color-coded grid.
  3. plot_value_heatmap   : State-value heatmap for a fixed battery/mask slice.
  4. plot_multi_heatmaps  : Side-by-side heatmaps varying battery level.
  5. plot_trajectory      : Animate/display the greedy policy trajectory on the grid.

Color scheme for cell types:
  S (Start)    : Light blue
  F (Free)     : White / light grey
  D (Danger)   : Red
  R (Rescue)   : Green
  C (Charging) : Blue
  W (Wind)     : Yellow
  X (Blocked)  : Dark grey
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from dp.env import DroneRescueEnv


# Cell type to background colour mapping
CELL_COLORS = {
    "S": "#AED6F1",   # Light blue — Start
    "F": "#F8F9FA",   # Near-white — Free/Safe
    "D": "#E74C3C",   # Red — Danger
    "R": "#27AE60",   # Green — Rescue
    "C": "#2980B9",   # Blue — Charging
    "W": "#F39C12",   # Orange/Yellow — Wind
    "X": "#717D7E",   # Dark grey — Blocked
}

# Arrow symbols for policy directions
DIRECTION_SYMBOLS = {
    "UP":    "↑",
    "DOWN":  "↓",
    "LEFT":  "←",
    "RIGHT": "→",
    "HOVER": "●",
}


def _draw_colored_grid(ax, env: DroneRescueEnv, rescue_mask: int = None):
    """Draw the color-coded grid background on a matplotlib Axes object.

    Fills each cell with a background colour matching its type, and adds the
    cell-type symbol as a text overlay.

    Args:
        ax: Matplotlib Axes to draw on.
        env (DroneRescueEnv): Environment containing grid layout information.
        rescue_mask (int, optional): Current rescue bitmask to show which
                                     targets have been collected (shown as 'F').
    """
    n = env.grid_size

    for r in range(n):
        for c in range(n):
            # Determine cell type, accounting for rescued targets
            cell_type = env.grid[r, c]
            if rescue_mask is not None and cell_type == "R":
                pos = (r, c)
                if pos in env.rescue_index:
                    bit = 1 << env.rescue_index[pos]
                    if not (rescue_mask & bit):
                        cell_type = "F"   # Target has been rescued → free cell

            # Fill cell with type-specific background colour
            color = CELL_COLORS.get(cell_type, "#FFFFFF")
            rect = plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                facecolor=color, edgecolor="#BBBBBB", linewidth=0.8
            )
            ax.add_patch(rect)

            # Add cell type label in the corner (small, light text)
            ax.text(
                c - 0.38, r - 0.38, cell_type,
                ha="left", va="top", fontsize=7,
                color="#444444", alpha=0.7
            )


def plot_grid_layout(env: DroneRescueEnv):
    """Plot the static grid layout showing all cell types with a color legend.

    Draws the full grid with colored backgrounds, cell-type symbols, and a
    legend explaining each color. Includes grid coordinates on the axes.

    Args:
        env (DroneRescueEnv): Environment to visualise.
    """
    n = env.grid_size
    fig, ax = plt.subplots(figsize=(7, 7))

    # Draw the coloured grid
    _draw_colored_grid(ax, env)

    # Add coordinate labels in each cell
    for r in range(n):
        for c in range(n):
            ax.text(c, r, f"({r},{c})", ha="center", va="center",
                    fontsize=8, color="#222222", fontweight="bold")

    # Axis formatting
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)   # Invert y-axis so (0,0) is top-left
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"Col {i}" for i in range(n)], fontsize=9)
    ax.set_yticklabels([f"Row {i}" for i in range(n)], fontsize=9)
    ax.set_title(
        f"Drone Rescue Grid Layout — Group 91 ({n}×{n})\n"
        f"Battery: {env.max_battery} | Wind Prob: {env.wind_prob:.0%} | Max Steps: {env.max_steps}",
        fontsize=12, fontweight="bold", pad=12
    )

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=CELL_COLORS[t], label=f"{t} — {name}")
        for t, name in [("S","Start"), ("F","Free/Safe"), ("D","Danger"),
                        ("R","Rescue"), ("C","Charging"), ("W","Wind"), ("X","Blocked")]
    ]
    ax.legend(handles=legend_patches, loc="upper right", bbox_to_anchor=(1.32, 1.0),
              fontsize=9, framealpha=0.95)

    plt.tight_layout()
    plt.show()


def plot_policy(
    env: DroneRescueEnv,
    policy: Dict,
    battery: int,
    rescue_mask: int,
    step: int = 0,
    title: str = None,
    ax=None,
):
    """Plot the greedy policy arrows overlaid on the colour-coded grid.

    For each non-terminal, non-blocked cell, draws the optimal action as a
    directional arrow (or hover dot). Blocked and terminal cells are shown
    with their base cell colour but no arrow.

    Args:
        env (DroneRescueEnv): Environment with grid layout.
        policy (Dict): Optimal policy mapping state → action string.
        battery (int): Battery level to query the policy for (fixed slice).
        rescue_mask (int): Rescue bitmask to query the policy for.
        step (int): Step count for state lookup. Default 0.
        title (str, optional): Custom plot title. Auto-generated if None.
        ax: Optional existing Axes to draw on. Creates a new figure if None.
    """
    n = env.grid_size
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Draw color-coded grid background
    _draw_colored_grid(ax, env, rescue_mask=rescue_mask)

    # Overlay policy arrows for each cell
    for r in range(n):
        for c in range(n):
            state = (r, c, battery, rescue_mask, step)
            if state in policy:
                action = policy[state]
                symbol = DIRECTION_SYMBOLS.get(action, "?")
                ax.text(
                    c, r, symbol,
                    ha="center", va="center",
                    fontsize=22, fontweight="bold",
                    color="#1A1A2E",
                )
            elif env.grid[r, c] == "X":
                # Blocked cell: show X
                ax.text(c, r, "✕", ha="center", va="center",
                        fontsize=20, color="#FFFFFF")

    # Axis formatting
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"C{i}" for i in range(n)], fontsize=9)
    ax.set_yticklabels([f"R{i}" for i in range(n)], fontsize=9)

    if title is None:
        rescued_n = bin(~rescue_mask & ((1 << len(env.rescue_positions)) - 1)).count("1")
        title = (f"Optimal Policy — Battery={battery}, "
                 f"Rescued={rescued_n}/{len(env.rescue_positions)}")
    ax.set_title(title, fontsize=11, fontweight="bold")

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_value_heatmap(
    env: DroneRescueEnv,
    value: np.ndarray,
    state_to_index: Dict,
    battery: int,
    rescue_mask: int,
    step: int = 0,
    title: str = None,
    ax=None,
):
    """Plot a heatmap of V*(s) for a fixed battery level and rescue mask.

    Produces a colour-gradient heatmap where brighter colours indicate
    higher state values. Numeric values are annotated in each cell.
    Blocked and unreachable cells are shown in grey.

    Args:
        env (DroneRescueEnv): Environment with grid layout.
        value (np.ndarray): Optimal value array (indexed by state_to_index).
        state_to_index (Dict): Maps state tuple → index into value array.
        battery (int): Fixed battery level for the slice.
        rescue_mask (int): Fixed rescue mask for the slice.
        step (int): Fixed step for the slice. Default 0.
        title (str, optional): Plot title. Auto-generated if None.
        ax: Optional existing Axes. Creates new figure if None.
    """
    n = env.grid_size
    heatmap = np.full((n, n), np.nan, dtype=float)

    # Populate the heatmap grid with V*(s) values for this slice
    for r in range(n):
        for c in range(n):
            if env.grid[r, c] == "X":
                continue   # Skip blocked cells
            state = (r, c, battery, rescue_mask, step)
            if state in state_to_index:
                heatmap[r, c] = value[state_to_index[state]]

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 7))

    # Draw the heatmap image
    im = ax.imshow(
        heatmap, cmap="RdYlGn", origin="upper",
        vmin=np.nanmin(heatmap) if not np.all(np.isnan(heatmap)) else -1,
        vmax=np.nanmax(heatmap) if not np.all(np.isnan(heatmap)) else 1,
    )

    # Annotate each cell with its V* value and cell type
    for r in range(n):
        for c in range(n):
            cell = env.grid[r, c]
            if np.isnan(heatmap[r, c]):
                # Blocked or unreachable: grey overlay
                ax.add_patch(plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    facecolor="#888888", edgecolor="black", linewidth=0.5
                ))
                ax.text(c, r, cell, ha="center", va="center",
                        fontsize=12, color="white")
            else:
                # Annotate with V* value
                ax.text(c, r, f"{heatmap[r, c]:.1f}", ha="center", va="center",
                        fontsize=10, fontweight="bold", color="#1A1A1A")

    if standalone:
        plt.colorbar(im, ax=ax, label="State Value V*(s)")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"C{i}" for i in range(n)], fontsize=9)
    ax.set_yticklabels([f"R{i}" for i in range(n)], fontsize=9)

    if title is None:
        title = f"V*(s) Heatmap — Battery={battery}, Mask={bin(rescue_mask)}"
    ax.set_title(title, fontsize=11, fontweight="bold")

    if standalone:
        plt.tight_layout()
        plt.show()

    return im


def plot_multi_heatmaps(
    env: DroneRescueEnv,
    value: np.ndarray,
    state_to_index: Dict,
    rescue_mask: int,
    battery_levels: List[int] = None,
):
    """Plot side-by-side value heatmaps for multiple battery levels.

    Allows comparison of how state values change as battery depletes.
    Each sub-plot shows V*(s) for a different battery level, with the
    same rescue mask.

    Args:
        env (DroneRescueEnv): Environment with grid layout.
        value (np.ndarray): Optimal value array.
        state_to_index (Dict): State → index mapping.
        rescue_mask (int): Fixed rescue bitmask for all sub-plots.
        battery_levels (List[int], optional): Battery levels to show.
                                              Defaults to [5, 10, max].
    """
    if battery_levels is None:
        # Sample low, medium, and full battery levels
        battery_levels = [5, 10, env.max_battery]

    n_plots = len(battery_levels)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 7))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(
        f"State-Value Analysis: V*(s) vs Battery Level (Group 91)\n"
        f"Rescue Mask = {bin(rescue_mask)} ({bin(rescue_mask).count('1')} targets remaining)",
        fontsize=13, fontweight="bold"
    )

    ims = []
    for ax, battery in zip(axes, battery_levels):
        im = plot_value_heatmap(
            env, value, state_to_index,
            battery=battery,
            rescue_mask=rescue_mask,
            title=f"Battery = {battery}",
            ax=ax,
        )
        ims.append(im)

    # Shared colorbar
    fig.colorbar(ims[-1], ax=axes, label="State Value V*(s)", shrink=0.6)
    plt.tight_layout()
    plt.show()


def plot_trajectory(
    env: DroneRescueEnv,
    trajectory: List[Tuple],
    title: str = "Greedy Policy Trajectory",
):
    """Visualise the greedy trajectory as a sequence of numbered steps on the grid.

    Draws the drone's path as a directed line with numbered waypoints.
    Each step is annotated with the action taken and the step number.

    Args:
        env (DroneRescueEnv): Environment with grid layout.
        trajectory (List[tuple]): Output of extract_greedy_trajectory().
                                   Each element: (state, action, reward).
        title (str): Plot title.
    """
    n = env.grid_size
    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw the base grid (using initial rescue_mask from the first state)
    initial_mask = trajectory[0][0][3] if trajectory else (1 << len(env.rescue_positions)) - 1
    _draw_colored_grid(ax, env, rescue_mask=initial_mask)

    # Extract path positions
    positions = [step[0][:2] for step in trajectory]   # (row, col) sequence

    # Draw the trajectory as connected arrows
    for i in range(len(positions) - 1):
        r1, c1 = positions[i]
        r2, c2 = positions[i + 1]
        ax.annotate(
            "",
            xy=(c2, r2), xytext=(c1, r1),
            arrowprops=dict(
                arrowstyle="->",
                color="#1A1A2E",
                lw=2.0,
                connectionstyle="arc3,rad=0.0"
            )
        )

    # Number each waypoint
    for i, (r, c) in enumerate(positions):
        ax.text(
            c + 0.15, r - 0.15, str(i),
            ha="left", va="top",
            fontsize=8, color="#C0392B", fontweight="bold"
        )

    # Mark start (S) and end
    if positions:
        rs, cs = positions[0]
        ax.plot(cs, rs, "go", markersize=12, label="Start", zorder=5)
        re, ce = positions[-1]
        ax.plot(ce, re, "r*", markersize=14, label="End", zorder=5)

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"C{i}" for i in range(n)], fontsize=9)
    ax.set_yticklabels([f"R{i}" for i in range(n)], fontsize=9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=9)

    # Compute total reward from trajectory
    total_reward = sum(step[2] for step in trajectory)
    ax.text(
        0.02, 0.02,
        f"Steps: {len(trajectory)-1} | Total Reward: {total_reward:.1f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()
    plt.show()


def plot_policy_multi_slice(
    env: DroneRescueEnv,
    policy: Dict,
    rescue_mask: int,
    battery_levels: List[int] = None,
):
    """Plot policy visualisations for multiple battery levels side by side.

    Useful for showing how the optimal policy changes as battery depletes —
    for example, the drone may prefer charging stations at low battery.

    Args:
        env (DroneRescueEnv): Environment with grid layout.
        policy (Dict): Optimal policy mapping state → action.
        rescue_mask (int): Rescue bitmask to fix for all slices.
        battery_levels (List[int], optional): Battery levels to show.
    """
    if battery_levels is None:
        battery_levels = [5, 10, env.max_battery]

    n_plots = len(battery_levels)
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 7))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle(
        "Optimal Policy π*(s) — Policy Varies with Battery Level (Group 91)",
        fontsize=13, fontweight="bold"
    )

    for ax, battery in zip(axes, battery_levels):
        plot_policy(
            env, policy, battery=battery,
            rescue_mask=rescue_mask,
            title=f"Battery = {battery}",
            ax=ax
        )

    plt.tight_layout()
    plt.show()
