"""
Reusable Multi-Armed Bandit (MAB) Environment Wrapper (Group 91).

This module defines the MABEnvironment class that simulates the clinical trial
bandit problem. The environment tracks which medicine (arm) is assigned to each
patient, the binary clinical outcome, and the utility score used as the reward.

Key design decisions:
  - Hidden success probabilities are fixed per medicine (arm) and unknown to algorithms.
  - Each call to step() advances exactly one patient (one time step).
  - Arms are tracked with pulls, successes, and total accumulated reward.
  - Utility score (not raw outcome) is used to compute cumulative reward, as it
    accounts for patient severity.
"""

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from mab.dataset import compute_reward


class MABEnvironment:
    """Bandit environment that tracks arm pulls, rewards, and patient assignments.

    This environment models a hospital clinical trial where each medicine is an
    arm with a fixed but hidden success probability. At each time step (patient),
    the algorithm selects an arm (medicine). The environment samples the clinical
    outcome stochastically based on the arm's hidden probability, computes the
    utility reward (penalised for high severity), and records everything in the
    dataset.

    Attributes:
        hidden_probs (List[float]): True success probabilities for each arm.
        dataset (pd.DataFrame): Patient dataset with outcome columns.
        n_arms (int): Number of available medicines (arms).
        pulls (np.ndarray): Count of how many times each arm has been pulled.
        total_rewards (np.ndarray): Accumulated utility reward per arm.
        successes (np.ndarray): Count of successful (recovered) outcomes per arm.
        current_step (int): Index of the current patient being treated.
    """

    def __init__(self, hidden_probs: List[float], dataset: pd.DataFrame):
        """Initialise the MAB environment with given arm probabilities and patient data.

        Args:
            hidden_probs (List[float]): Hidden success probability P_i for each arm i.
            dataset (pd.DataFrame): Patient dataset created by create_patient_dataset().
        """
        self.hidden_probs = hidden_probs
        self.dataset = dataset.copy()   # Work on a copy to avoid mutating the original
        self.n_arms = len(hidden_probs)
        self.reset()

    def reset(self):
        """Reset the environment to its initial state for a fresh algorithm run.

        Clears all assignment columns back to placeholders and resets all arm
        statistics (pulls, rewards, successes) and the step counter. Must be called
        before starting any new strategy simulation.
        """
        # Reset dataset assignment columns
        self.dataset["assigned_medicine"] = -1
        self.dataset["clinical_outcome"] = -1
        self.dataset["utility_score"] = 0.0

        # Reset arm-level statistics
        self.pulls = np.zeros(self.n_arms, dtype=int)           # Pull counts per arm
        self.total_rewards = np.zeros(self.n_arms, dtype=float) # Accumulated reward
        self.successes = np.zeros(self.n_arms, dtype=int)       # Recovery counts
        self.current_step = 0                                    # Patient index

    def step(self, arm: int) -> Tuple[int, float]:
        """Assign the selected medicine (arm) to the current patient and get reward.

        Samples a binary clinical outcome from the arm's hidden Bernoulli distribution,
        computes the severity-adjusted utility score, updates arm statistics, and
        records the results in the dataset.

        Args:
            arm (int): Index of the selected medicine (0 to n_arms-1).

        Returns:
            tuple:
                - outcome (int): 1 if the patient recovered, 0 otherwise.
                - reward (float): Utility score = outcome × (1 - 0.1 × severity).
        """
        # Fetch current patient's severity score
        patient = self.dataset.loc[self.current_step]
        severity = int(patient["severity_score"])

        # Sample clinical outcome from the arm's hidden Bernoulli distribution
        outcome = int(np.random.rand() < self.hidden_probs[arm])

        # Compute severity-adjusted utility reward (lower reward for severe patients)
        reward = compute_reward(outcome, severity)

        # Update arm-level statistics
        self.pulls[arm] += 1
        self.total_rewards[arm] += reward
        self.successes[arm] += outcome

        # Record this step's results into the dataset
        self.dataset.at[self.current_step, "assigned_medicine"] = arm
        self.dataset.at[self.current_step, "clinical_outcome"] = outcome
        self.dataset.at[self.current_step, "utility_score"] = reward

        self.current_step += 1
        return outcome, reward

    def get_average_rewards(self) -> np.ndarray:
        """Compute average utility reward per arm across all pulls so far.

        Arms that have never been pulled return 0.0 (safe division via numpy).

        Returns:
            np.ndarray: Average reward for each arm (shape: [n_arms]).
        """
        return np.divide(
            self.total_rewards,
            self.pulls,
            out=np.zeros_like(self.total_rewards),
            where=self.pulls > 0,
        )

    def get_success_rates(self) -> np.ndarray:
        """Compute empirical success rate (recovery proportion) per arm.

        Returns:
            np.ndarray: Empirical recovery rate for each arm (shape: [n_arms]).
        """
        return np.divide(
            self.successes.astype(float),
            self.pulls,
            out=np.zeros_like(self.total_rewards),
            where=self.pulls > 0,
        )

    def get_cumulative_reward(self) -> float:
        """Return the total accumulated utility reward across all treated patients.

        Returns:
            float: Sum of utility_score column for all steps taken so far.
        """
        return float(self.dataset["utility_score"].sum())

    def get_current_step(self) -> int:
        """Return the index of the next patient to be treated.

        Returns:
            int: Current step index (0-indexed patient ID).
        """
        return int(self.current_step)

    def get_arm_summary(self) -> Dict[int, dict]:
        """Return a per-arm summary of pulls, success rate, and average reward.

        Useful for post-run analysis to see which arm dominated selection.

        Returns:
            dict: Keys are arm indices; values are dicts with 'pulls',
                  'success_rate', 'avg_reward'.
        """
        avg_rewards = self.get_average_rewards()
        success_rates = self.get_success_rates()
        summary = {}
        for i in range(self.n_arms):
            summary[i] = {
                "pulls": int(self.pulls[i]),
                "success_rate": round(float(success_rates[i]), 4),
                "avg_reward": round(float(avg_rewards[i]), 4),
            }
        return summary
