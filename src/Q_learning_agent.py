import numpy as np
import random  
import numpy as np
from typing import Dict


class QLearningAgent:
    def __init__(self, 
                 actions: int,                        # Number of possible actions
                 gamma: float = 1.0                   # Discount factor (0 ≤ γ ≤ 1)
                 ) -> None:
        """
        Q-Learning Agent 
        
        Parameters:
        actions (int): Number of possible actions
        gamma (float): Discount factor (0 ≤ γ ≤ 1)
        """
        self.q_table: Dict[int, np.ndarray] = {}  # State -> Q-values mapping
        self.max_actions: int = actions + 1       # Maximum number of actions
        self.alpha: float = 1.0                   # Learning rate (0 < α ≤ 1)
        self.epsilon: float = 1.0                 # Exploration rate (0 < ε < 1)
        self.gamma: float = gamma                 # Discount factor
        self.s: int = 0                          # Lower threshold for (s,S) policy
        self.S: int = 0                          # Upper threshold for (s,S) policy

    def _init_state(self, state: int) -> None:
        """
        Initialize a state with  Q-values = 0 if it doesn't exist.
        
        Parameters:
        state (int): State to initialize in Q-table
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.max_actions)

    def choose_action(self, state: int) -> int:
        """
        Epsilon-greedy action selection.
        
        Parameters:
        state (int): Current state
        
        Returns:
        int: Selected action index
        """
        self._init_state(state)  # Ensure state exists in Q-table
        
        if random.uniform(0, 1) < self.epsilon:
            # Exploration: random action
            return np.random.randint(0, self.max_actions)  # high value is exclusive
        else:
            # Exploitation: greedy action
            return np.argmax(self.q_table[state])

    def policy_evaluate_action_generate(self, 
                                       state: int, 
                                       s: int, 
                                       S: int
                                       ) -> int:
        """
        Generate action based on (s,S) inventory policy evaluation.
        
        Parameters:
        state (int): Current inventory state
        s (int): Lower threshold - reorder point
        S (int): Upper threshold - order up to level
        
        Returns:
        int: Action (order quantity) based on input(s,S) policy
        """
        self._init_state(state)
        self.s = s
        self.S = S
        
        if state < s:
            return S - state  # Order up to S
        else:
            return 0          # No order

    def policy_evaluate(self, 
                       state: int, 
                       action: int, 
                       reward: float, 
                       next_state: int, 
                       done: bool
                       ) -> None:
        """
        Q-value update using policy evaluation with (s,S) policy.
        
        Parameters:
        state (int): Current state
        action (int): Action taken
        reward (float): Reward received
        next_state (int): Next state reached
        done (bool): Whether episode is terminated
        """
        self._init_state(state)  # Initialize current state if needed
        
        # Initialize next_state if it's not terminal
        if not done:
            self._init_state(next_state)
            
        if action >= self.max_actions:
            print(f"action {action}")
            print("index error")
        
        # Get next action based on policy
        next_action: int = self.policy_evaluate_action_generate(next_state, self.s, self.S)
        
        current_q: float = self.q_table[state][action]
        next_q: float = self.q_table[next_state][next_action] if not done else 0.0
        new_q: float = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        
        # Update all actions for this state (state value, not state action value)
        for action_idx in range(self.max_actions):
            self.q_table[state][action_idx] = new_q

    def learn(self, 
              state: int, 
              action: int, 
              reward: float, 
              next_state: int, 
              done: bool
              ) -> None:
        """
        Standard Q-learning update using Bellman equation.
        
        Parameters:
        state (int): Current state
        action (int): Action taken
        reward (float): Reward received
        next_state (int): Next state reached
        done (bool): Whether episode is terminated
        """
        self._init_state(state)  # Initialize current state if needed
        
        # Initialize next_state if it's not terminal
        if not done:
            self._init_state(next_state)
            
        if action >= self.max_actions:
            print(f"action {action}")
            print("index error")
        
        current_q: float = self.q_table[state][action]
        max_next_q: float = np.max(self.q_table[next_state]) if not done else 0.0
        new_q: float = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q