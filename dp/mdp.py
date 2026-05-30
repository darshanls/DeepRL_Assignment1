"""
Dynamic Programming Solution for Drone Rescue Environment (Group 91).

This module implements Value Iteration to compute the optimal value function
V*(s) and optimal policy π*(s) for the DroneRescueEnv MDP.

Value Iteration Algorithm:
  Repeatedly apply the Bellman optimality update until convergence:

    V(s) ← max_a Σ_{s'} P(s'|s,a) × [R(s,a,s') + γ × V(s')]

  Where:
    - P(s'|s,a) : Transition probability (from get_transitions)
    - R(s,a,s') : Immediate reward
    - γ          : Discount factor (default 1.0 for undiscounted)
    - θ          : Convergence threshold (1e-3 as required by assignment)

Convergence is declared when the maximum change in V(s) across all states
in one iteration falls below θ = 10⁻³.
"""

import time
from typing import Dict, List, Tuple
import numpy as np
from dp.env import DroneRescueEnv


def value_iteration(
    env: DroneRescueEnv,
    reachable_states: List[Tuple[int, int, int, int, int]],
    theta: float = 1e-3,
    gamma: float = 1.0,
    max_iters: int = 2000,
) -> Tuple[np.ndarray, Dict[Tuple[int, int, int, int, int], str], int, float, List[float]]:
    """Compute the optimal value function and policy using Value Iteration.

    Implements the standard Value Iteration algorithm (Bellman optimality
    operator applied until convergence) over the set of reachable states.

    Stochastic transitions from wind zones are handled correctly via
    env.get_transitions(state, action), which returns the full distribution
    P(s'|s,a) with associated rewards.

    Stopping criterion:
        Converges when max_s |V_new(s) - V_old(s)| < θ = 10⁻³.

    Args:
        env (DroneRescueEnv): The drone rescue environment instance.
        reachable_states (List[tuple]): All reachable states from BFS enumeration.
        theta (float): Convergence threshold. Default 1e-3 (assignment required).
        gamma (float): Discount factor γ ∈ (0, 1]. Default 1.0 (undiscounted).
        max_iters (int): Maximum iterations before forced stopping.

    Returns:
        tuple:
            - value (np.ndarray): Optimal value V*(s) for each reachable state.
                                   Indexed by state_to_index mapping.
            - policy (Dict): Maps each non-terminal state → optimal action string.
            - iterations (int): Number of iterations until convergence.
            - final_delta (float): Max |V_new - V_old| at the last iteration.
            - delta_history (List[float]): Per-iteration delta for convergence plot.
    """
    # Build index mapping: state tuple → array position
    state_to_index = {state: idx for idx, state in enumerate(reachable_states)}

    # Initialise value function to zero for all states
    value = np.zeros(len(reachable_states), dtype=float)

    policy = {}         # Optimal action per non-terminal state
    delta_history = []  # Track convergence for plotting

    for iteration in range(1, max_iters + 1):
        delta = 0.0   # Maximum value change this iteration

        # --- Bellman optimality update for every non-terminal state ---
        for state in reachable_states:
            if env.is_terminal(state):
                # Terminal states have zero value (no future reward possible)
                continue

            best_value  = -np.inf
            best_action = env.ACTIONS[0]   # Default fallback

            # --- Evaluate all valid actions using Bellman expectation ---
            for action in env.valid_actions(state):
                # Get full stochastic transition distribution for (state, action)
                transitions = env.get_transitions(state, action)

                # Compute expected value: Σ_s' P(s'|s,a) × [R(s,a,s') + γ × V(s')]
                action_value = 0.0
                for prob, next_state, reward, done in transitions:
                    # Get value of next state (0 if terminal or not indexed)
                    next_idx = state_to_index.get(next_state, None)
                    if next_idx is not None:
                        next_value = value[next_idx]
                    else:
                        next_value = 0.0   # Unvisited states treated as zero value
                    # Bellman update contribution from this transition outcome
                    action_value += prob * (reward + gamma * next_value)

                # Track the best action value
                if action_value > best_value:
                    best_value  = action_value
                    best_action = action

            # Compute delta: maximum change in value function
            idx = state_to_index[state]
            delta = max(delta, abs(value[idx] - best_value))

            # Update value and policy for this state
            value[idx]    = best_value
            policy[state] = best_action

        # Record delta for convergence plot
        delta_history.append(delta)

        # --- Check convergence criterion ---
        if delta < theta:
            return value, policy, iteration, delta, delta_history

    # If max iterations reached without convergence, return best result so far
    return value, policy, max_iters, delta, delta_history


def extract_greedy_trajectory(
    env: DroneRescueEnv,
    policy: Dict[Tuple[int, int, int, int, int], str],
    max_steps: int = None,
) -> List[Tuple[Tuple[int, int, int, int, int], str, float]]:
    """Simulate a greedy trajectory by following the optimal policy deterministically.

    Useful for visualising the learned policy as a sequence of states, actions,
    and rewards from the start state.

    Args:
        env (DroneRescueEnv): The environment (used for transitions).
        policy (Dict): Optimal policy mapping state → action.
        max_steps (int, optional): Maximum steps to simulate.
                                    Defaults to env.max_steps.

    Returns:
        List of (state, action, reward) triples for each step taken.
    """
    if max_steps is None:
        max_steps = env.max_steps

    trajectory = []
    state = env.reset()   # Start from the initial state

    for _ in range(max_steps):
        if env.is_terminal(state):
            break   # Stop when the episode ends

        # Choose action from optimal policy; default to HOVER if state not in policy
        action = policy.get(state, "HOVER")

        # Step deterministically (use the most probable outcome for visualization)
        transitions = env.get_transitions(state, action)
        # Pick the highest-probability transition
        best_t = max(transitions, key=lambda t: t[0])
        _, next_state, reward, done = best_t

        trajectory.append((state, action, reward))
        state = next_state

    # Append final state with no action
    trajectory.append((state, None, 0.0))
    return trajectory
