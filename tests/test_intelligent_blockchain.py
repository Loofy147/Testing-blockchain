import unittest
import sys
import time
import numpy as np

sys.path.insert(0, '.')

from blockchain.blockchain import (
    IntelligentBlockchain,
    Transaction,
    NetworkMetrics,
    IntelligentFeeMarket,
    IntelligentMempool
)

class TestAdvancedIntelligentBlockchain(unittest.TestCase):

    def setUp(self):
        # Suppress print output during tests
        import builtins
        self._original_print = builtins.print
        builtins.print = lambda *args, **kwargs: None

        self.blockchain = IntelligentBlockchain(difficulty=1)
        self.blockchain.balances['Alice'] = 1000.0  # Pre-fund Alice

    def tearDown(self):
        # Restore print output
        import builtins
        builtins.print = self._original_print

    def test_network_metrics(self):
        metrics = NetworkMetrics()
        self.assertEqual(metrics.get_congestion_score(), 0.0)

        start_time = time.time()

        # Phase 1: Low velocity (10 tx over ~20 seconds) -> ~0.5 tx/s
        for i in range(10):
            metrics.record_transaction(Transaction("A", "B", 1, 0.1, start_time - 40 + i * 2))

        # Phase 2: High velocity (10 tx over ~10 seconds) -> ~1.0 tx/s
        for i in range(10):
            metrics.record_transaction(Transaction("A", "B", 1, 0.1, start_time - 20 + i))

        # The most recent 10 transactions (Phase 2) have a higher rate than the 10 before them (Phase 1)
        self.assertGreater(metrics.get_velocity_trend(), 0)

        # Add more transactions to test congestion score
        for i in range(5):
             metrics.record_transaction(Transaction("A", "B", 1, 0.1, start_time))

        self.assertAlmostEqual(metrics.get_congestion_score(), 0.25)

    def test_intelligent_fee_market(self):
        market = self.blockchain.fee_market
        metrics = self.blockchain.metrics

        # Low load
        rec = market.calculate_dynamic_fee(5, metrics, 0.5)
        self.assertIn('recommended_fee', rec)
        self.assertIn('price_tiers', rec)
        self.assertGreater(rec['price_tiers']['priority'], rec['price_tiers']['standard'])

        # High load
        for _ in range(50):
            metrics.record_transaction(Transaction("A", "B", 1, 0.1, time.time()))

        high_load_rec = market.calculate_dynamic_fee(50, metrics, 0.8)
        self.assertGreater(high_load_rec['recommended_fee'], rec['recommended_fee'])

        # Test learning
        initial_base_fee = market.base_fee
        for _ in range(20):
            market.learn_from_outcome(Transaction("A", "B", 1, 0.01, time.time()), False, 60)
        market.calculate_dynamic_fee(10, metrics) # Trigger learning update
        self.assertGreater(market.base_fee, initial_base_fee)

    def test_intelligent_mempool(self):
        mempool = self.blockchain.mempool
        tx1 = Transaction("A", "B", 10, 0.5, time.time()) # High fee/amount ratio
        tx2 = Transaction("C", "D", 100, 0.2, time.time()) # Low fee/amount ratio

        mempool.add_transaction(tx1)
        mempool.add_transaction(tx2)

        self.assertGreater(tx1.priority_score, tx2.priority_score)

        optimal_set = mempool.get_optimal_transactions(max_count=1)
        self.assertEqual(len(optimal_set), 1)
        self.assertEqual(optimal_set[0].sender, "A")

        # Test anti-dominance
        for i in range(5):
            mempool.add_transaction(Transaction("A", f"Z{i}", 1, 1.0, time.time()))

        fair_set = mempool.get_optimal_transactions(max_count=5)
        sender_a_count = sum(1 for tx in fair_set if tx.sender == "A")
        self.assertLessEqual(sender_a_count, 3)

    def test_add_transaction_insufficient_funds(self):
        result = self.blockchain.add_transaction("Alice", "Bob", 2000)
        self.assertFalse(result['success'])
        self.assertIn('error', result)

    def test_full_mining_cycle(self):
        self.assertEqual(len(self.blockchain.chain), 1)

        # Add some transactions
        self.blockchain.add_transaction("Alice", "Bob", 50)
        self.blockchain.add_transaction("Alice", "Charlie", 20)

        self.assertEqual(len(self.blockchain.mempool.pending), 2)

        mined_block = self.blockchain.mine_block("MinerX")

        self.assertIsNotNone(mined_block)
        self.assertEqual(len(self.blockchain.chain), 2)
        self.assertEqual(len(self.blockchain.mempool.pending), 0)

        # Check miner reward (base + fees)
        total_fees = sum(tx.fee for tx in mined_block.transactions if tx.sender != "network")
        reward_tx = next(tx for tx in mined_block.transactions if tx.sender == "network")

        self.assertEqual(reward_tx.recipient, "MinerX")
        self.assertAlmostEqual(reward_tx.amount, self.blockchain.mining_reward + total_fees)

        # Check balances after mining
        self.assertLess(self.blockchain.get_balance("Alice"), 1000)
        self.assertEqual(self.blockchain.get_balance("Bob"), 50)


if __name__ == '__main__':
    unittest.main(verbosity=2)
