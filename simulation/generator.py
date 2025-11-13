import random
import time
import numpy as np
import sys

sys.path.insert(0, '.')

from blockchain.blockchain import Transaction

class TransactionGenerator:
    """
    Generates synthetic but realistic transaction loads for simulation.
    """

    def __init__(self, user_accounts: list, blockchain):
        if not user_accounts:
            raise ValueError("User accounts cannot be empty.")
        self.user_accounts = user_accounts
        self.blockchain = blockchain

    def generate_transactions(self, scenario: str, duration: int) -> list:
        """
        Generates a stream of transactions based on a predefined scenario.

        Args:
            scenario (str): The load scenario ('stable', 'spike', 'wave').
            duration (int): The number of simulation steps (e.g., seconds) to run.

        Returns:
            list: A list of generated Transaction objects.
        """
        if scenario == 'stable':
            return self._generate_stable_load(duration)
        elif scenario == 'spike':
            return self._generate_spike_load(duration)
        elif scenario == 'wave':
            return self._generate_wave_load(duration)
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

    def _generate_stable_load(self, duration: int, tx_per_step: int = 5):
        """Generates a consistent, low-level stream of transactions."""
        transactions = []
        print(f"--- Generating STABLE load for {duration} steps ({tx_per_step} tx/step)...")
        for i in range(duration):
            for _ in range(tx_per_step):
                sender, recipient = random.sample(self.user_accounts, 2)
                amount = round(random.uniform(0.1, 10), 2)
                urgency = round(random.uniform(0.1, 0.5), 1)

                # Check balance before creating transaction
                if self.blockchain.get_balance(sender) > amount * 2: # Ensure enough for tx + fee
                    result = self.blockchain.add_transaction(sender, recipient, amount, urgency=urgency)
                    if result['success']:
                        transactions.append(result['transaction'])
            time.sleep(0.01) # Simulate passage of time
        return transactions

    def _generate_spike_load(self, duration: int, spike_time: int = 1, spike_magnitude: int = 50):
        """Generates a sudden, massive burst of transactions."""
        transactions = []
        print(f"--- Generating SPIKE load for {duration} steps (spike at step {spike_time})...")
        for i in range(duration):
            if i == spike_time:
                print(f"*** SIMULATING SPIKE ({spike_magnitude} transactions) ***")
                for _ in range(spike_magnitude):
                    sender, recipient = random.sample(self.user_accounts, 2)
                    amount = round(random.uniform(1, 50), 2)
                    urgency = round(random.uniform(0.7, 1.0), 1)
                    if self.blockchain.get_balance(sender) > amount * 2:
                        result = self.blockchain.add_transaction(sender, recipient, amount, urgency=urgency)
                        if result['success']:
                            transactions.append(result['transaction'])
            else:
                # Normal low-level traffic
                if random.random() < 0.5:
                    sender, recipient = random.sample(self.user_accounts, 2)
                    amount = round(random.uniform(0.1, 5), 2)
                    if self.blockchain.get_balance(sender) > amount * 2:
                         result = self.blockchain.add_transaction(sender, recipient, amount)
                         if result['success']:
                            transactions.append(result['transaction'])
            time.sleep(0.01)
        return transactions

    def _generate_wave_load(self, duration: int, cycles: int = 2):
        """Generates a load that oscillates between high and low."""
        transactions = []
        print(f"--- Generating WAVE load for {duration} steps ({cycles} cycles)...")
        for i in range(duration):
            # Use a sine wave to determine the number of transactions per step
            tx_per_step = int((np.sin(2 * np.pi * cycles * i / duration) + 1.1) * 5)

            for _ in range(tx_per_step):
                sender, recipient = random.sample(self.user_accounts, 2)
                amount = round(random.uniform(0.5, 20), 2)
                urgency = round(random.uniform(0.3, 0.8), 1)
                if self.blockchain.get_balance(sender) > amount * 2:
                    result = self.blockchain.add_transaction(sender, recipient, amount, urgency=urgency)
                    if result['success']:
                        transactions.append(result['transaction'])
            time.sleep(0.01)
        return transactions
