import numpy as np
import random

class ProtocolOptimizerRL:
    """
    A Q-learning agent to dynamically adjust blockchain parameters like difficulty.
    """

    def __init__(self, actions: list, learning_rate: float = 0.1, discount_factor: float = 0.9, exploration_rate: float = 1.0, exploration_decay: float = 0.995):
        self.actions = actions  # e.g., [-1, 0, 1] for decrease, keep, increase difficulty
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = 0.01

        self.q_table = {}  # Q-table: state -> [q_value_for_action_1, q_value_for_action_2, ...]

    def get_state(self, network_metrics) -> tuple:
        """
        Discretizes the continuous network metrics into a manageable state for the Q-table.
        """
        # Discretize avg_block_time: e.g., <8s, 8-12s, >12s
        avg_block_time = network_metrics.get_average_block_time()
        if avg_block_time < 8:
            time_state = 0 # Too fast
        elif avg_block_time <= 12:
            time_state = 1 # Ideal
        else:
            time_state = 2 # Too slow

        # Discretize congestion: e.g., <30%, 30-70%, >70%
        congestion = network_metrics.get_congestion_score()
        if congestion < 0.3:
            congestion_state = 0 # Low
        elif congestion <= 0.7:
            congestion_state = 1 # Medium
        else:
            congestion_state = 2 # High

        return (time_state, congestion_state)

    def choose_action(self, state: tuple) -> int:
        """
        Chooses an action using an epsilon-greedy policy.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            # Exploit: choose the best known action
            q_values = self.q_table.get(state, np.zeros(len(self.actions)))
            return self.actions[np.argmax(q_values)]

    def update_q_table(self, state: tuple, action: int, reward: float, next_state: tuple):
        """
        Updates the Q-value for a given state-action pair using the Bellman equation.
        """
        action_index = self.actions.index(action)

        old_value = self.q_table.get(state, np.zeros(len(self.actions)))[action_index]

        next_max = np.max(self.q_table.get(next_state, np.zeros(len(self.actions))))

        # Q-learning formula
        new_value = old_value + self.lr * (reward + self.gamma * next_max - old_value)

        # Update the Q-table
        if state not in self.q_table:
            self.q_table[state] = np.zeros(len(self.actions))

        new_q_values = self.q_table[state].copy()
        new_q_values[action_index] = new_value
        self.q_table[state] = new_q_values

    def decay_exploration(self):
        """Reduces the exploration rate over time."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    @staticmethod
    def calculate_reward(metrics, target_block_time: float = 10.0) -> float:
        """
        Calculates a reward based on how close the average block time is to the target.
        A higher reward is given for being closer to the target time.
        """
        avg_block_time = metrics.get_average_block_time()

        # Reward is inversely proportional to the squared error from the target
        error = abs(avg_block_time - target_block_time)
        reward = 1.0 / (1.0 + error**2)

        # Penalize heavily if the network is either too fast or completely stalled
        if avg_block_time < target_block_time / 2 or avg_block_time > target_block_time * 2:
            reward -= 1.0

        return reward
