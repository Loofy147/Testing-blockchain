import unittest
import sys
import time

# Add the project root to the Python path
sys.path.insert(0, '.')

from blockchain.blockchain import Blockchain, Transaction, FeePredictor

class TestIntelligentBlockchain(unittest.TestCase):

    def setUp(self):
        """Set up a new blockchain for each test."""
        # Suppress print output during tests
        import builtins
        self._original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None

        self.blockchain = Blockchain(difficulty=1) # Use low difficulty for speed
        # Pre-fund an account for testing transactions by mining a block
        self.blockchain.add_transaction("network", "Alice", 1000.0)
        self.blockchain.mine_pending_transactions("network_miner")


    def tearDown(self):
        # Restore print output
        import builtins
        builtins.print = self._original_print

    def test_transaction_creation(self):
        """Test valid and invalid transaction creation."""
        # Valid transaction
        tx = Transaction("Alice", "Bob", 50, 0.1)
        self.assertEqual(tx.sender, "Alice")
        self.assertEqual(tx.amount, 50)

        # Invalid: negative amount
        with self.assertRaises(ValueError):
            Transaction("Alice", "Bob", -50, 0.1)

        # Invalid: negative fee
        with self.assertRaises(ValueError):
            Transaction("Alice", "Bob", 50, -0.1)

    def test_fee_predictor(self):
        """Test the AI fee predictor."""
        predictor = FeePredictor()
        predictor.train_model()

        # Test prediction (should be clamped to min_fee)
        low_congestion_fee = predictor.predict_fee(0)
        self.assertEqual(low_congestion_fee, predictor.min_fee)

        # Test prediction at a known point - adjust expectation based on linear regression
        high_congestion_fee = predictor.predict_fee(50)
        self.assertAlmostEqual(high_congestion_fee, 1.672, places=3)

        # Test max clamping
        extreme_congestion_fee = predictor.predict_fee(1000)
        self.assertEqual(extreme_congestion_fee, predictor.max_fee)

    def test_add_transaction_success(self):
        """Test adding a valid transaction."""
        # Get Alice's balance after initial funding
        self.blockchain.balances = {} # Clear cache to be sure
        alice_initial_balance = self.blockchain.get_balance("Alice")
        self.assertEqual(alice_initial_balance, 1000.0)

        result = self.blockchain.add_transaction("Alice", "Bob", 100)
        self.assertTrue(result)
        self.assertEqual(len(self.blockchain.pending_transactions), 1)
        tx = self.blockchain.pending_transactions[0]
        self.assertEqual(tx.sender, "Alice")
        self.assertGreater(tx.fee, 0) # Fee should be auto-calculated

    def test_add_transaction_insufficient_funds(self):
        """Test adding a transaction with insufficient balance."""
        result = self.blockchain.add_transaction("Alice", "Bob", 2000) # Alice only has 1000
        self.assertFalse(result)
        self.assertEqual(len(self.blockchain.pending_transactions), 0)

    def test_mining_block_and_rewards(self):
        """Test the mining process, including rewards and fees."""
        self.blockchain.add_transaction("Alice", "Bob", 50, fee=0.5)
        self.blockchain.add_transaction("Alice", "Charlie", 20, fee=0.2)

        mined_block = self.blockchain.mine_pending_transactions("MinerX")

        self.assertIsNotNone(mined_block)
        self.assertEqual(len(self.blockchain.chain), 3) # Genesis, Funding, this one
        self.assertEqual(len(self.blockchain.pending_transactions), 0)

        # Verify miner's reward (base reward + total fees)
        expected_reward = self.blockchain.mining_reward + 0.5 + 0.2
        reward_tx = mined_block.transactions[-1]
        self.assertEqual(reward_tx.recipient, "MinerX")
        self.assertEqual(reward_tx.amount, expected_reward)

    def test_get_balance(self):
        """Test balance calculation after several transactions."""
        # Initial state
        self.assertEqual(self.blockchain.get_balance("Alice"), 1000.0)

        # Add transactions
        self.blockchain.add_transaction("Alice", "Bob", 100, fee=1.0)
        self.blockchain.add_transaction("Alice", "Charlie", 50, fee=0.5)
        self.blockchain.mine_pending_transactions("MinerX")

        # Recalculate balances from the chain
        self.blockchain.balances = {} # Clear cache

        # Alice's balance should be reduced by amounts + fees
        expected_alice_balance = 1000.0 - (100 + 1.0) - (50 + 0.5)
        self.assertAlmostEqual(self.blockchain.get_balance("Alice"), expected_alice_balance)

        # Bob and Charlie should have received their amounts
        self.assertAlmostEqual(self.blockchain.get_balance("Bob"), 100.0)
        self.assertAlmostEqual(self.blockchain.get_balance("Charlie"), 50.0)

        # Miner should have the reward
        miner_reward = self.blockchain.mining_reward + 1.0 + 0.5
        self.assertAlmostEqual(self.blockchain.get_balance("MinerX"), miner_reward)

    def test_chain_validity(self):
        """Test the integrity of the blockchain."""
        self.assertTrue(self.blockchain.is_chain_valid())

        self.blockchain.add_transaction("Alice", "Bob", 10)
        self.blockchain.mine_pending_transactions("MinerX")
        self.assertTrue(self.blockchain.is_chain_valid())

        # Tamper with a transaction
        self.blockchain.chain[2].transactions[0].amount = 9999
        self.assertFalse(self.blockchain.is_chain_valid()) # Hash mismatch

        # Fix the transaction but tamper with the hash linkage
        self.blockchain.chain[2].transactions[0].amount = 10
        # Don't recalculate hash, just tamper with linkage to test that check
        self.blockchain.chain[1].hash = "tampered_hash"
        self.assertFalse(self.blockchain.is_chain_valid()) # Previous hash mismatch

if __name__ == '__main__':
    unittest.main(verbosity=2)
