"""
Custom Drone Rescue Environment for DP Assignment (Group 91).

This module implements the DroneRescueEnv class — a fully custom grid-world
environment modelling an autonomous drone rescuing civilians in a disaster zone.
The environment is formulated as a finite Markov Decision Process (MDP).

Grid Configuration (Group 91, last digit = 1, so 5×5 grid):
  - Grid size    : 5×5
  - Max battery  : 15 (last digit odd)
  - Wind prob    : 0.20 (last digit 0–4)
  - Max steps    : 50
  - Rescue targets  : 2
  - Charging stations : 1
  - Danger zones  : 3
  - Blocked cells : 2

State Representation:
  Each state is a 5-tuple: (row, col, battery, rescue_mask, step)
    - row, col     : Drone position in the grid (0-indexed)
    - battery      : Remaining battery units [0, max_battery]
    - rescue_mask  : Integer bitmask encoding which rescue targets remain active
                     (bit i = 1 means target i has NOT yet been rescued)
    - step         : Number of actions taken so far in this episode

Cell Type Symbols:
  S = Start (top-left corner, (0,0))
  F = Free / Safe cell
  D = Danger zone (-10 reward on entry)
  R = Rescue target (+20 reward on entry, disappears after rescue)
  C = Charging station (+5 on entry; battery refilled; hover +2)
  W = Wind zone (stochastic movement)
  X = Blocked cell / obstacle (cannot enter)

Reward Structure:
  Rescue target reached  : +20
  Enter danger zone      : -10
  Battery exhausted      : -20
  Reach charging station : +5
  Regular movement       : -1
  Hover on non-charger   : -1
  Hover on charger       : +0 net (battery +2, cost -1 already applied)
"""

from typing import List, Tuple, Dict
import numpy as np
from parameters import GROUP_NUMBER


class DroneRescueEnv:
    """Grid-world MDP environment for the autonomous drone rescue problem.

    The drone starts at position (0,0) with full battery and must rescue all
    civilian targets while managing battery, avoiding danger zones, and using
    charging stations before power depletes.

    Public API (required by assignment):
        reset()          → Returns the initial state tuple.
        step(action)     → Takes an action string, returns (next_state, reward, done).
        render()         → Prints the current grid with the drone's position.
        get_transitions  → Returns the stochastic transition distribution for DP.
        valid_actions    → Returns valid actions for a given state.
        is_terminal      → Returns True if the given state is a terminal state.
        enumerate_reachable_states → BFS over all states reachable from start.
    """

    # Action set: five possible drone actions
    ACTIONS = ["UP", "DOWN", "LEFT", "RIGHT", "HOVER"]

    # Grid movement deltas for directional actions
    DELTA: Dict[str, Tuple[int, int]] = {
        "UP":    (-1,  0),
        "DOWN":  ( 1,  0),
        "LEFT":  ( 0, -1),
        "RIGHT": ( 0,  1),
    }

    # Reward values as specified in the assignment
    REWARD_RESCUE   =  20    # Reaching a rescue target
    REWARD_DANGER   = -10    # Entering a danger zone
    REWARD_BATTERY  = -20    # Battery exhausted (reaches 0)
    REWARD_CHARGING =   5    # Reaching a charging station (entry only)
    REWARD_MOVE     =  -1    # Regular movement or hover cost

    def __init__(self, group_id: int = GROUP_NUMBER):
        """Initialise the drone rescue environment for a given group ID.

        All grid parameters (size, battery, wind probability, cell placements)
        are derived deterministically from the group ID per assignment rules.

        Args:
            group_id (int): Team group number. Used to derive all environment
                            parameters. Default: GROUP_NUMBER from parameters.py.
        """
        self.group_id = group_id
        last_digit = group_id % 10

        # --- Derive environment parameters from group ID ---
        # Grid size: 5×5 if last digit 0–4, else 6×6
        self.grid_size = 6 if last_digit >= 5 else 5

        # Max battery: 15 if last digit odd, else 10
        self.max_battery = 15 if last_digit % 2 == 1 else 10

        # Wind probability: 0.30 if last digit 5–9, else 0.20
        self.wind_prob = 0.30 if last_digit >= 5 else 0.20

        # Max episode steps: 75 for 6×6 grid, 50 for 5×5 grid
        self.max_steps = 75 if self.grid_size == 6 else 50

        # Starting position is always the top-left corner
        self.start = (0, 0)

        # Build the grid and record cell positions
        self._build_grid()

        # Initialise internal simulation state (for step() / render() API)
        self._current_state = None
        self.reset()

    def _build_grid(self):
        """Construct the grid layout and record positions for each cell type.

        Grid placements are deterministic based on grid_size (group-derived).
        The 5×5 layout is used for Group 91 (last digit = 1).

        5×5 Grid Layout (Group 91):
            (0,0)S  (0,1)F  (0,2)F  (0,3)F  (0,4)D
            (1,0)F  (1,1)X  (1,2)W  (1,3)R  (1,4)F
            (2,0)F  (2,1)F  (2,2)C  (2,3)F  (2,4)X
            (3,0)F  (3,1)R  (3,2)F  (3,3)D  (3,4)W
            (4,0)F  (4,1)D  (4,2)F  (4,3)F  (4,4)F

        Symbols: S=Start, F=Free, D=Danger, R=Rescue, C=Charger, W=Wind, X=Blocked
        """
        n = self.grid_size
        self.grid = np.full((n, n), "F", dtype="<U1")
        self.grid[self.start] = "S"   # Start is always top-left

        if self.grid_size == 6:
            # 6×6 layout for groups with last digit 5–9
            self.rescue_positions   = [(1, 4), (3, 1), (4, 5)]
            self.charging_positions = [(2, 2), (5, 3)]
            self.danger_positions   = [(0, 4), (3, 3), (4, 1), (5, 0)]
            self.wind_positions     = [(1, 2), (2, 4), (4, 3)]
            self.blocked_positions  = [(1, 1), (2, 5), (4, 2)]
        else:
            # 5×5 layout for Group 91 (last digit = 1, which is in 0–4 range)
            self.rescue_positions   = [(1, 3), (3, 1)]   # 2 rescue targets
            self.charging_positions = [(2, 2)]            # 1 charging station
            self.danger_positions   = [(0, 4), (3, 3), (4, 1)]  # 3 danger zones
            self.wind_positions     = [(1, 2), (3, 4)]   # 2 wind zones
            self.blocked_positions  = [(1, 1), (2, 4)]   # 2 blocked cells

        # Write cell symbols onto the grid array
        for pos in self.rescue_positions:
            self.grid[pos] = "R"
        for pos in self.charging_positions:
            self.grid[pos] = "C"
        for pos in self.danger_positions:
            self.grid[pos] = "D"
        for pos in self.wind_positions:
            self.grid[pos] = "W"
        for pos in self.blocked_positions:
            self.grid[pos] = "X"

        # Build a lookup from rescue position → bit index for bitmask encoding
        self.rescue_index: Dict[Tuple[int, int], int] = {
            pos: idx for idx, pos in enumerate(self.rescue_positions)
        }

    # =========================================================
    # PUBLIC API
    # =========================================================

    def reset(self) -> Tuple[int, int, int, int, int]:
        """Reset the environment to the initial state for a new episode.

        Sets the drone back to position (0,0), restores full battery, marks
        all rescue targets as active (bitmask = all 1s), and resets step count.

        Returns:
            tuple: Initial state (row=0, col=0, battery=max_battery,
                   rescue_mask=all_active, step=0).
        """
        # Bitmask with all rescue targets active: bit i=1 for each target i
        initial_mask = (1 << len(self.rescue_positions)) - 1

        # Construct the initial state tuple
        self._current_state = (
            self.start[0],   # row
            self.start[1],   # col
            self.max_battery, # battery starts full
            initial_mask,    # all rescue targets active
            0,               # step counter starts at 0
        )
        return self._current_state

    def step(self, action: str) -> Tuple[Tuple[int, int, int, int, int], float, bool]:
        """Execute an action from the current state and advance one time step.

        Applies stochastic wind dynamics if the drone is on a wind cell and
        the action is a directional move. Otherwise uses deterministic dynamics.

        Args:
            action (str): One of 'UP', 'DOWN', 'LEFT', 'RIGHT', 'HOVER'.

        Returns:
            tuple:
                - next_state (tuple): New state after action.
                - reward (float): Immediate reward for this transition.
                - done (bool): True if the episode has ended.

        Raises:
            ValueError: If called in a terminal state.
        """
        if self.is_terminal(self._current_state):
            raise ValueError("Cannot call step() on a terminal state. Call reset() first.")

        # Sample a transition from the stochastic distribution
        transitions = self.get_transitions(self._current_state, action)

        # Sample one outcome according to transition probabilities
        probs    = np.array([t[0] for t in transitions])
        outcomes = list(range(len(transitions)))
        sampled  = np.random.choice(outcomes, p=probs)

        prob, next_state, reward, done = transitions[sampled]
        self._current_state = next_state
        return next_state, reward, done

    def render(self, state: Tuple[int, int, int, int, int] = None):
        """Print a text representation of the grid with the drone's current position.

        Overlays the drone marker 'D' on the grid at the drone's (row, col).
        Also prints state information (battery, rescued targets, step count).

        Args:
            state (tuple, optional): State to render. Defaults to the current
                                     internal state set by step() / reset().
        """
        if state is None:
            state = self._current_state

        row, col, battery, rescue_mask, step = state

        # Build a display grid: copy base grid, overlay drone and rescued cells
        display = self.grid.copy()

        # Mark rescued targets as free cells (they disappear after rescue)
        for pos, idx in self.rescue_index.items():
            if not (rescue_mask & (1 << idx)):   # Bit is 0 → already rescued
                display[pos] = "F"

        # Place drone marker (overrides whatever cell it's on)
        display[row, col] = "🚁"

        # Cell symbol legend
        legend = {
            "S": "Start", "F": "Free", "D": "Danger", "R": "Rescue",
            "C": "Charger", "W": "Wind", "X": "Blocked", "🚁": "Drone",
        }

        # Print the grid
        print("+" + "----+" * self.grid_size)
        for r in range(self.grid_size):
            row_str = "|"
            for c in range(self.grid_size):
                cell = display[r, c]
                if cell == "🚁":
                    row_str += " 🚁 |"
                else:
                    row_str += f"  {cell} |"
            print(row_str)
            print("+" + "----+" * self.grid_size)

        # Print state summary below the grid
        rescued_count = bin(~rescue_mask & ((1 << len(self.rescue_positions)) - 1)).count("1")
        remaining     = bin(rescue_mask).count("1")
        print(f"  Battery: {battery}/{self.max_battery}  |  "
              f"Step: {step}/{self.max_steps}  |  "
              f"Rescued: {rescued_count}/{len(self.rescue_positions)}  |  "
              f"Remaining targets: {remaining}")
        print()

    # =========================================================
    # TRANSITION DYNAMICS
    # =========================================================

    def get_transitions(
        self,
        state: Tuple[int, int, int, int, int],
        action: str,
    ) -> List[Tuple[float, Tuple[int, int, int, int, int], float, bool]]:
        """Return the full stochastic transition distribution for a (state, action) pair.

        For wind zones (W), a directional action has a probability of wind_prob
        of being redirected to a uniformly random direction (Up/Down/Left/Right).
        The intended action succeeds with probability (1 - wind_prob).

        For all other cells and for HOVER, transitions are deterministic.

        This method is the primary interface for Dynamic Programming (Value Iteration),
        as DP requires the full distribution P(s'|s,a) with associated rewards.

        Args:
            state (tuple): Current state (row, col, battery, rescue_mask, step).
            action (str): Action from ACTIONS list.

        Returns:
            List of (prob, next_state, reward, done) tuples.
            Probabilities sum to 1.0. Stochastic outcomes are merged if they
            lead to the same next_state (probabilities accumulated).
        """
        if self.is_terminal(state):
            # Terminal states: self-loop with zero reward and done=True
            return [(1.0, state, 0.0, True)]

        row, col, battery, rescue_mask, step = state

        # --- Check if the drone is currently on a wind cell ---
        on_wind_cell = (self.grid[row, col] == "W")

        if not on_wind_cell or action == "HOVER":
            # Deterministic: single outcome with probability 1.0
            next_state, reward = self._compute_next_state(state, action)
            done = self.is_terminal(next_state)
            return [(1.0, next_state, reward, done)]

        # --- Stochastic wind dynamics ---
        # The wind may redirect any directional move to a random direction.
        # Intended action: probability (1 - wind_prob)
        # Each of 4 random directions: probability wind_prob / 4

        raw_transitions = {}

        def add_outcome(prob, actual_action):
            """Helper: compute next state for an actual action and accumulate probabilities."""
            ns, rew = self._compute_next_state(state, actual_action)
            done    = self.is_terminal(ns)
            key     = ns   # Use next_state as the key for merging
            if key not in raw_transitions:
                raw_transitions[key] = [0.0, rew, done]
            raw_transitions[key][0] += prob
            # Note: rewards for the same next_state are identical (they only
            # depend on next_state), so no conflict in using the first one.

        # Intended action succeeds with (1 - wind_prob)
        add_outcome(1.0 - self.wind_prob, action)

        # Wind disturbance: redirect to each of 4 directions uniformly
        random_prob = self.wind_prob / 4.0
        for random_action in ["UP", "DOWN", "LEFT", "RIGHT"]:
            add_outcome(random_prob, random_action)

        # Build final list of (prob, next_state, reward, done) tuples
        return [(data[0], ns, data[1], data[2]) for ns, data in raw_transitions.items()]

    def _compute_next_state(
        self,
        state: Tuple[int, int, int, int, int],
        action: str,
    ) -> Tuple[Tuple[int, int, int, int, int], float]:
        """Compute the deterministic next state and reward for a (state, action) pair.

        Handles all environment dynamics:
          - Battery consumption (every action costs 1 unit).
          - Hover on charging station: battery += 2 (bonus, net cost = -1 already applied).
          - Entry into charging station: battery refills to max.
          - Rescue target collection: target disappears, rescue_mask updated.
          - Blocked cell collision: drone stays, battery still consumed.
          - Boundary collision: drone stays, battery still consumed.
          - Danger zone penalty.
          - Battery exhaustion penalty.

        Args:
            state (tuple): Current state (row, col, battery, rescue_mask, step).
            action (str): Action string from ACTIONS.

        Returns:
            tuple: (next_state, reward)
        """
        row, col, battery, rescue_mask, step = state

        # Every action (move or hover) consumes 1 battery unit
        battery -= 1

        if action == "HOVER":
            # Hovering: drone stays in place
            new_row, new_col = row, col
            if self.grid[row, col] == "C":
                # On a charging station: hovering adds +2 battery (capped at max)
                battery = min(self.max_battery, battery + 2)
        else:
            # Directional move: compute intended new position
            new_row, new_col = self._move(row, col, action)

            # If landing on a charging station: refill battery to maximum
            if (new_row, new_col) in self.charging_positions:
                battery = self.max_battery

        # Update rescue bitmask: if drone lands on an active rescue target, collect it
        new_mask = rescue_mask
        if self._is_rescue_active(new_row, new_col, rescue_mask):
            # Clear the bit for this rescue target (mark as rescued)
            bit = 1 << self.rescue_index[(new_row, new_col)]
            new_mask = rescue_mask & ~bit

        # --- Compute reward for this transition ---
        reward = self.REWARD_MOVE   # Base cost: -1 for every action

        # Danger zone: additional -10 penalty (does NOT terminate the episode)
        if self.grid[new_row, new_col] == "D":
            reward += self.REWARD_DANGER

        # Charging station entry reward (only on entry, not hovering in place)
        if (new_row, new_col) in self.charging_positions and (new_row, new_col) != (row, col):
            reward += self.REWARD_CHARGING

        # Rescue target collected: +20 reward
        if self._is_rescue_active(new_row, new_col, rescue_mask):
            reward += self.REWARD_RESCUE

        # Battery exhausted: additional -20 penalty and episode ends
        battery = max(battery, 0)   # Clamp battery at 0 (no negative battery)
        if battery <= 0:
            reward += self.REWARD_BATTERY

        new_state = (new_row, new_col, battery, new_mask, step + 1)
        return new_state, reward

    def _move(self, row: int, col: int, action: str) -> Tuple[int, int]:
        """Apply a directional action and return the resulting grid position.

        If the move would leave the grid boundary or enter a blocked cell (X),
        the drone remains in its current position (bounce-back behaviour).

        Args:
            row (int): Current row.
            col (int): Current column.
            action (str): Directional action ('UP', 'DOWN', 'LEFT', 'RIGHT').

        Returns:
            tuple: (new_row, new_col) after applying the action.
        """
        dr, dc = self.DELTA[action]
        new_row, new_col = row + dr, col + dc

        # Check grid boundary
        if not (0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size):
            return row, col   # Bounce back to current position

        # Check for blocked cell obstacle
        if self.grid[new_row, new_col] == "X":
            return row, col   # Bounce back; action still costs 1 battery unit

        return new_row, new_col

    def _is_rescue_active(
        self, row: int, col: int, rescue_mask: int
    ) -> bool:
        """Check if a rescue target at position (row, col) is still active.

        Uses bitmask encoding: bit i=1 means rescue target i has NOT been rescued.

        Args:
            row (int): Grid row to check.
            col (int): Grid column to check.
            rescue_mask (int): Current rescue bitmask.

        Returns:
            bool: True if there is an active (unrescued) target at (row, col).
        """
        if (row, col) not in self.rescue_index:
            return False
        bit = 1 << self.rescue_index[(row, col)]
        return (rescue_mask & bit) != 0

    # =========================================================
    # STATE SPACE UTILITIES
    # =========================================================

    def is_terminal(self, state: Tuple[int, int, int, int, int]) -> bool:
        """Determine if a state is terminal (episode ends).

        An episode terminates when any of the following occur:
          1. Battery reaches 0 (drone is stranded).
          2. All rescue targets have been rescued (rescue_mask == 0).
          3. Maximum step limit is exceeded.

        Args:
            state (tuple): State to evaluate.

        Returns:
            bool: True if the episode has terminated.
        """
        _, _, battery, rescue_mask, step = state
        all_rescued = (rescue_mask == 0)   # All bits cleared → all targets saved
        return battery <= 0 or all_rescued or step >= self.max_steps

    def valid_actions(self, state: Tuple[int, int, int, int, int]) -> List[str]:
        """Return the list of valid actions available from a given state.

        All 5 actions are always valid from non-terminal states. The environment
        handles boundary and blocked-cell collisions internally (drone stays in
        place), so no actions need to be filtered at this level.

        Args:
            state (tuple): State to query.

        Returns:
            List[str]: All ACTIONS if state is non-terminal, else [].
        """
        if self.is_terminal(state):
            return []   # No actions available in terminal states
        return self.ACTIONS.copy()

    def enumerate_reachable_states(self) -> List[Tuple[int, int, int, int, int]]:
        """Enumerate all states reachable from the initial state via BFS.

        Uses stochastic transitions (get_transitions) to ensure that wind-zone
        states are properly explored. This guarantees the DP algorithm has
        complete coverage of the state space.

        Returns:
            List[tuple]: All reachable states as (row, col, battery, mask, step).
        """
        from collections import deque

        # Start state: top-left, full battery, all targets active, step=0
        start_mask  = (1 << len(self.rescue_positions)) - 1
        start_state = (self.start[0], self.start[1], self.max_battery, start_mask, 0)

        queue   = deque([start_state])
        visited = {start_state}

        while queue:
            state = queue.popleft()
            if self.is_terminal(state):
                continue   # Do not expand terminal states

            for action in self.valid_actions(state):
                # Use get_transitions to include stochastic wind outcomes
                for _prob, next_state, _reward, _done in self.get_transitions(state, action):
                    if next_state not in visited:
                        visited.add(next_state)
                        queue.append(next_state)

        return list(visited)

    def state_description(self) -> str:
        """Return a human-readable description of the state representation.

        Returns:
            str: Multi-line explanation of each component of the state tuple.
        """
        n_targets = len(self.rescue_positions)
        n_states_approx = (
            self.grid_size ** 2
            * (self.max_battery + 1)
            * (2 ** n_targets)
            * (self.max_steps + 1)
        )
        return (
            f"State = (row, col, battery, rescue_mask, step)\n"
            f"  row         : Drone row position [0, {self.grid_size-1}]\n"
            f"  col         : Drone column position [0, {self.grid_size-1}]\n"
            f"  battery     : Remaining battery [0, {self.max_battery}]\n"
            f"  rescue_mask : Bitmask — {n_targets} bits, one per rescue target\n"
            f"                bit i=1 → target i still active, bit i=0 → rescued\n"
            f"                Range: [0, {(1 << n_targets) - 1}]\n"
            f"  step        : Actions taken so far [0, {self.max_steps}]\n"
            f"\n"
            f"  Theoretical upper bound on states:\n"
            f"    {self.grid_size}×{self.grid_size} × {self.max_battery+1} battery levels "
            f"× {2**n_targets} rescue masks × {self.max_steps+1} steps\n"
            f"    = {n_states_approx:,} states (most are unreachable)\n"
        )
