"""
Bandit Strategy Implementations for MAB Assignment (Group 91).

This module implements three exploration-exploitation strategies for the
Multi-Armed Bandit clinical trial problem:

  1. Greedy (Immediate Exploitation): Test each arm briefly, then always exploit
     the best-performing arm. No further exploration after the initial phase.

  2. Epsilon-Greedy (Controlled Exploration): With probability ε explore a random
     arm; otherwise exploit the current best arm. Supports ε ∈ {0.01, 0.10, 0.50}.

  3. UCB1 (Confidence-Based Selection): Select arms based on their upper confidence
     bound — initially favouring under-explored arms; confidence shrinks as
     evidence grows, naturally balancing exploration and exploitation.

Each strategy returns:
  - The updated MABEnvironment (with dataset populated).
  - A history dictionary containing per-step records for analysis.
"""

from typing import Dict, List, Tuple
import numpy as np
from mab.environment import MABEnvironment


# Type alias for strategy result histories
StrategyHistory = Dict[str, List]


def greedy_strategy(
    env: MABEnvironment,
    initial_pulls: int = 10,
) -> Tuple[MABEnvironment, StrategyHistory]:
    """Run the Immediate Exploitation (Greedy) strategy over all 1000 patients.

    Strategy logic (Task 2):
      Phase 1 – Exploration: Each arm is pulled exactly `initial_pulls` times in
                round-robin order, cycling through arm 0, 1, ..., K-1. This gives
                the algorithm minimal evidence to estimate each arm's effectiveness.
      Phase 2 – Exploitation: Once all arms have been tested `initial_pulls` times,
                the arm with the highest average utility reward is selected for ALL
                remaining patients. No further exploration occurs.

    This reflects the hospital administrator's "use the best known treatment" policy.

    Args:
        env (MABEnvironment): Bandit environment to simulate.
        initial_pulls (int): Number of initial pulls per arm before exploitation.
                             Default is 10 (so 60 exploration steps for 6 arms).

    Returns:
        tuple:
            - env (MABEnvironment): Environment with dataset fully populated.
            - history (dict): Contains:
                'cumulative_reward': list of cumulative reward after each step.
                'selected_arm': list of arm chosen at each step.
                'phase': list of 'explore' or 'exploit' tag per step.
    """
    history = {
        "cumulative_reward": [],
        "selected_arm": [],
        "phase": [],
    }
    env.reset()   # Clear previous run data before starting

    # Total exploration steps = number of arms × pulls per arm
    exploration_end = env.n_arms * initial_pulls

    for t in range(len(env.dataset)):
        if t < exploration_end:
            # --- Phase 1: Exploration ---
            # Cycle through arms sequentially: arm = t // initial_pulls
            # e.g., with initial_pulls=10 and 6 arms:
            #   t=0..9 → arm 0, t=10..19 → arm 1, ..., t=50..59 → arm 5
            arm = t // initial_pulls
            phase = "explore"
        else:
            # --- Phase 2: Exploitation ---
            # Always select the arm with the highest average utility reward
            arm = int(np.argmax(env.get_average_rewards()))
            phase = "exploit"

        # Execute the chosen arm on the current patient
        env.step(arm)

        # Record step-level history
        history["cumulative_reward"].append(env.get_cumulative_reward())
        history["selected_arm"].append(arm)
        history["phase"].append(phase)

    return env, history


def epsilon_greedy_strategy(
    env: MABEnvironment,
    epsilon: float,
    initial_pulls: int = 1,
) -> Tuple[MABEnvironment, StrategyHistory]:
    """Run the Epsilon-Greedy Controlled Exploration strategy over 1000 patients.

    Strategy logic (Task 3):
      At each time step, a Bernoulli draw with probability ε decides the action:
        - With probability ε  → Explore: select a uniformly random arm.
        - With probability 1-ε → Exploit: select the arm with the highest
                                  average utility reward so far.
      If any arm has never been pulled yet, it is selected first (ensures
      each arm gets at least one observation before exploitation begins).

    The ε parameter controls the exploration-exploitation tradeoff:
        ε = 0.01  → aggressive exploitation, minimal exploration
        ε = 0.10  → balanced (recommended clinical default)
        ε = 0.50  → heavy exploration, slow convergence

    Args:
        env (MABEnvironment): Bandit environment to simulate.
        epsilon (float): Exploration probability in [0, 1].
        initial_pulls (int): Min pulls per arm before pure greedy kicks in.
                             Default 1 ensures every arm is tried at least once.

    Returns:
        tuple:
            - env (MABEnvironment): Environment with dataset fully populated.
            - history (dict): Contains:
                'cumulative_reward': cumulative reward after each step.
                'selected_arm': arm chosen at each step.
                'action_type': 'explore' or 'exploit' for each step.
                'epsilon': constant epsilon value (recorded for reference).
    """
    history = {
        "cumulative_reward": [],
        "selected_arm": [],
        "action_type": [],
        "epsilon": epsilon,   # Store scalar, not per-step list
    }
    env.reset()   # Clear previous run data before starting

    for t in range(len(env.dataset)):
        # Ensure every arm is tried at least once before greedy exploitation
        if np.any(env.pulls == 0):
            # Find the first arm that hasn't been pulled yet
            arm = int(np.argmin(env.pulls))
            action_type = "init_explore"

        elif np.random.rand() < epsilon:
            # --- Exploration: random arm selection ---
            arm = int(np.random.choice(env.n_arms))
            action_type = "explore"

        else:
            # --- Exploitation: choose arm with highest average utility reward ---
            arm = int(np.argmax(env.get_average_rewards()))
            action_type = "exploit"

        # Execute the chosen arm on the current patient
        env.step(arm)

        # Record step-level history
        history["cumulative_reward"].append(env.get_cumulative_reward())
        history["selected_arm"].append(arm)
        history["action_type"].append(action_type)

    return env, history


def ucb1_strategy(env: MABEnvironment) -> Tuple[MABEnvironment, StrategyHistory]:
    """Run the UCB1 (Upper Confidence Bound) Confidence-Based strategy over 1000 patients.

    Strategy logic (Task 4):
      UCB1 selects the arm with the highest Upper Confidence Bound score:

          UCB1(i, t) = avg_reward(i) + sqrt( 2 * ln(t) / pulls(i) )

      Where:
        - avg_reward(i) : exploitation term — how good arm i appears to be.
        - sqrt(2*ln(t)/pulls(i)) : exploration bonus — shrinks as arm i is pulled
                                    more, ensuring under-explored arms are favoured.

      Interpretation for the senior physician (Task 4 scenario):
        - Arms with fewer pulls have a larger exploration bonus, giving them more
          chances initially.
        - As evidence grows for each arm, the bonus shrinks and exploitation
          dominates — the algorithm naturally converges to the best arm.

      Initialisation: pull each arm exactly once before applying UCB1 scores
      (prevents division by zero in the confidence term).

    Args:
        env (MABEnvironment): Bandit environment to simulate.

    Returns:
        tuple:
            - env (MABEnvironment): Environment with dataset fully populated.
            - history (dict): Contains:
                'cumulative_reward': cumulative reward after each step.
                'selected_arm': arm chosen at each step.
                'ucb_scores': UCB1 score vector at each step (for analysis).
    """
    history = {
        "cumulative_reward": [],
        "selected_arm": [],
        "ucb_scores": [],
    }
    env.reset()   # Clear previous run data before starting

    for t in range(len(env.dataset)):
        if t < env.n_arms:
            # --- Initialisation phase: pull each arm once to avoid zero pulls ---
            # Arm selected in round-robin: arm 0, 1, ..., K-1
            arm = t
            ucb_scores = None
        else:
            # --- UCB1 selection: pick arm maximising upper confidence bound ---
            avg = env.get_average_rewards()             # Exploitation term
            # Exploration bonus: sqrt(2 * ln(t+1) / pulls[i]) per arm
            confidence = np.sqrt(2.0 * np.log(t + 1) / env.pulls)
            ucb_scores = avg + confidence               # Combined UCB1 score

            # Select the arm with the highest UCB1 score
            arm = int(np.argmax(ucb_scores))

        # Execute the chosen arm on the current patient
        env.step(arm)

        # Record step-level history
        history["cumulative_reward"].append(env.get_cumulative_reward())
        history["selected_arm"].append(arm)
        history["ucb_scores"].append(
            ucb_scores.tolist() if ucb_scores is not None else None
        )

    return env, history
