import unittest
import sys

sys.path.insert(0, '.')

from simulation.generator import TransactionGenerator
from simulation.runner import SimulationRunner
from blockchain.blockchain import IntelligentBlockchain

class TestSimulationEnvironment(unittest.TestCase):

    def setUp(self):
        # Suppress print output during tests
        import builtins
        self._original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None

        self.blockchain = IntelligentBlockchain(difficulty=1)
        self.users = ["Alice", "Bob", "Charlie", "Dave", "Eve"]
        # Fund all users to ensure enough liquidity for spike tests
        for user in self.users:
            self.blockchain.add_transaction("network", user, 5000)
        self.blockchain.mine_block("network_miner")

    def tearDown(self):
        import builtins
        builtins.print = self._original_print

    def test_transaction_generator(self):
        generator = TransactionGenerator(self.users, self.blockchain)

        # Test stable load
        transactions = generator.generate_transactions('stable', duration=2)
        self.assertGreater(len(transactions), 1) # Expect at least one tx

        # Test spike load - ensure full duration is passed to generator
        # The spike happens at duration // 2, so need at least that long
        transactions = generator.generate_transactions('spike', duration=4)
        self.assertGreater(len(transactions), 20, "Spike load should generate a high volume of transactions")

    def test_simulation_runner(self):
        runner = SimulationRunner(scenario='stable', duration=5, difficulty=1)
        runner.run()

        # Check that the simulation ran and produced some results
        self.assertGreater(len(runner.history), 0)
        self.assertIn('block_index', runner.history[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
