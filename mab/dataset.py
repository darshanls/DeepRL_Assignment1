"""
Synthetic Patient Dataset for Multi-Armed Bandit Assignment (Group 91).

This module creates a reproducible synthetic patient-treatment dataset
following the exact rules from the assignment specification:
  - Group number G = 91 drives all derived parameters.
  - Both Python random and NumPy random seeds are set to G for reproducibility.
  - Patient severity scores cycle from 1 (mild) to 5 (critical).
  - Utility rewards scale with clinical outcome and decrease with higher severity.
"""

import random
import numpy as np
import pandas as pd
from parameters import GROUP_NUMBER, MAB_NUM_PATIENTS


def create_patient_dataset(group_number: int = GROUP_NUMBER, n_patients: int = MAB_NUM_PATIENTS):
    """Create the synthetic MAB patient dataset according to assignment rules.

    The dataset encodes 1000 patients, each with a severity score and placeholder
    columns (assigned_medicine, clinical_outcome, utility_score) that are populated
    dynamically during algorithm execution.

    Derivation rules (Section 1 of assignment):
      - K = (G mod 3) + 5         → number of medicines (arms)
      - P_i = 0.4 + ((G+i) mod 6) * 0.07  → hidden success probability per medicine
      - Severity = (patient_id mod 5) + 1  → ranges 1 (mild) to 5 (critical)

    Args:
        group_number (int): Team group number used as random seed and for deriving
                            all dataset parameters. Default is GROUP_NUMBER from
                            parameters.py (91).
        n_patients (int): Number of synthetic patients to generate (default 1000).

    Returns:
        tuple:
            - dataset (pd.DataFrame): Patient records with columns:
                patient_id, severity_score, assigned_medicine,
                clinical_outcome, utility_score.
            - metadata (dict): Keys 'group_number', 'n_medicines', 'hidden_probs'.
    """
    # --- Reproducibility: seed BOTH Python random and NumPy as required ---
    random.seed(group_number)         # Seed Python's built-in random module
    np.random.seed(group_number)      # Seed NumPy's random module

    # Rule 1.1: Number of medicines (arms)
    k_medicines = (group_number % 3) + 5

    # Rule 1.2: Hidden success probability for each medicine i in {0,...,K-1}
    hidden_probs = [
        0.4 + ((group_number + i) % 6) * 0.07
        for i in range(k_medicines)
    ]

    # Generate patient IDs and compute severity scores (Rule 1.3)
    patient_ids = np.arange(n_patients)
    # Severity cycles 1,2,3,4,5,1,2,... using modulo arithmetic
    severity_scores = (patient_ids % 5) + 1

    # Build the base dataset; algorithm columns start as placeholders (-1 / 0.0)
    dataset = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "severity_score": severity_scores,
            "assigned_medicine": -1,    # Populated during algorithm execution
            "clinical_outcome": -1,     # Binary: 1=Recovered, 0=Not Recovered
            "utility_score": 0.0,       # Reward used for cumulative reward tracking
        }
    )

    metadata = {
        "group_number": group_number,
        "n_medicines": k_medicines,
        "hidden_probs": hidden_probs,
    }
    return dataset, metadata


def compute_reward(clinical_outcome: int, severity_score: int) -> float:
    """Compute the utility reward for a patient based on outcome and disease severity.

    Formula (Rule 1.3):
        UtilityScore = clinical_outcome × (1 - severity / 10)

    Interpretation:
        - Recovered (outcome=1), severity=1 (mild)     → reward = 0.9
        - Recovered (outcome=1), severity=5 (critical) → reward = 0.5
        - Not recovered (outcome=0)                     → reward = 0.0

    Treatment benefit decreases for severely ill patients because clinical
    success has reduced real-world impact when baseline health is critical.

    Args:
        clinical_outcome (int): 1 if patient recovered, 0 otherwise.
        severity_score (int): Disease severity in range [1, 5].

    Returns:
        float: Utility reward value in [0.0, 0.9].
    """
    return float(clinical_outcome) * (1.0 - 0.1 * severity_score)
