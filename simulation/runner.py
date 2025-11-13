import sys
import time
import pandas as pd
import random

sys.path.insert(0, '.')

from blockchain.blockchain import IntelligentBlockchain
from simulation.generator import TransactionGenerator

class SimulationRunner:
    """
    Orchestrates a full blockchain simulation to test AI performance.
    """

    def __init__(self, scenario: str, duration: int, difficulty: int = 1):
        self.scenario = scenario
        self.duration = duration
        self.blockchain = IntelligentBlockchain(difficulty=difficulty)

        # Setup user accounts
        self.users = ["Alice", "Bob", "Charlie", "Dave", "Eve", "Frank"]
        self._bootstrap_balances()

        self.generator = TransactionGenerator(self.users, self.blockchain)
        self.history = []

    def _bootstrap_balances(self, initial_funds: float = 1000.0):
        """Give all users an initial balance to start with."""
        print("--- Bootstrapping user balances...")
        for user in self.users:
            self.blockchain.add_transaction("network", user, initial_funds)

        # Mine the funding block
        self.blockchain.mine_block("network_miner")
        print("--- Balances bootstrapped.\n")

    def run(self):
        """Runs the full simulation from transaction generation to mining."""
        print(f"🚀 STARTING SIMULATION: Scenario='{self.scenario}', Duration={self.duration} steps")

        start_time = time.time()

        # Main simulation loop
        for step in range(self.duration):
            print(f"\n--- Step {step + 1}/{self.duration} ---")

            # 1. Generate new transactions for this step
            self.generator.generate_transactions(self.scenario, duration=1) # Generate for 1 step

            # 2. Mine a block if there are enough pending transactions
            if len(self.blockchain.mempool.pending) >= 5:
                miner = random.choice(self.users)
                mined_block = self.blockchain.mine_block(miner)

                if mined_block:
                    self._record_step_metrics(mined_block)

            # 3. Print AI insights periodically
            if (step + 1) % 10 == 0 and self.blockchain.protocol_optimizer: # Only print if AI is enabled
                self.blockchain.print_ai_insights()

        end_time = time.time()
        print(f"\n✅ SIMULATION COMPLETE. Total runtime: {end_time - start_time:.2f}s")

        self.report_results()

    def _record_step_metrics(self, block):
        """Record key metrics after each block is mined."""
        record = {
            'timestamp': time.time(),
            'block_index': block.index,
            'tx_in_block': len(block.transactions),
            'total_fees': block.total_fees,
            'avg_priority': block.avg_priority,
            'pending_tx': len(self.blockchain.mempool.pending),
            'base_fee': self.blockchain.fee_market.base_fee,
            'congestion': self.blockchain.metrics.get_congestion_score()
        }
        self.history.append(record)

    def report_results(self):
        """Generate and print a summary of the simulation results."""
        if not self.history:
            print("\n--- No blocks were mined during the simulation. ---")
            return

        df = pd.DataFrame(self.history)
        print("\n" + "="*60)
        print("📊 SIMULATION RESULTS")
        print("="*60)

        print("\nOverall Performance:")
        print(f"  Total Blocks Mined: {df['block_index'].max()}")
        print(f"  Total Transactions Processed: {df['tx_in_block'].sum()}")
        print(f"  Total Fees Collected: {df['total_fees'].sum():.4f}")

        print("\nFee Market Dynamics:")
        print(f"  Average Base Fee: {df['base_fee'].mean():.4f}")
        print(f"  Final Base Fee: {df['base_fee'].iloc[-1]:.4f}")

        print("\nNetwork State:")
        print(f"  Average Congestion: {df['congestion'].mean():.2%}")

        # Optional: Save to CSV for further analysis
        # df.to_csv(f"simulation_results_{self.scenario}.csv", index=False)
        # print(f"\nResults saved to simulation_results_{self.scenario}.csv")


if __name__ == "__main__":
    # To run a simulation: python simulation/runner.py <scenario> <duration>
    # Example: python simulation/runner.py spike 50
    if len(sys.argv) != 3:
        print("Usage: python simulation/runner.py <scenario> <duration>")
        print("Scenarios: stable, spike, wave")
        sys.exit(1)

    scenario = sys.argv[1]
    duration = int(sys.argv[2])

    runner = SimulationRunner(scenario=scenario, duration=duration)
    runner.run()
