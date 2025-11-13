import unittest
import sys
import numpy as np

sys.path.insert(0, '.')

from ai.fee_predictor_lstm import FeePredictorLSTM
from ai.protocol_optimizer_rl import ProtocolOptimizerRL
from blockchain.blockchain import NetworkMetrics

class TestAIComponents(unittest.TestCase):

    def test_fee_predictor_lstm(self):
        predictor = FeePredictorLSTM(n_features=2, n_steps=5)

        # Generate dummy time-series data
        history = [(i*0.1, i*0.01) for i in range(10)]

        # Test training
        predictor.train(history)
        self.assertTrue(predictor.is_trained)

        # Test prediction
        prediction = predictor.predict(history)
        self.assertIsInstance(prediction, float)
        self.assertGreater(prediction, 0)

    def test_protocol_optimizer_rl(self):
        optimizer = ProtocolOptimizerRL(actions=[-1, 0, 1])
        metrics = NetworkMetrics()

        # Test state discretization
        metrics.block_times.extend([9, 10, 11]) # Ideal time
        metrics.tx_history.extend([{'timestamp': 0, 'fee': 0, 'amount': 0}] * 50) # Medium congestion
        state = optimizer.get_state(metrics)
        self.assertEqual(state, (1, 1))

        # Test Q-table update
        action = 1
        reward = 0.8
        next_state = (1, 2)
        optimizer.update_q_table(state, action, reward, next_state)
        self.assertIn(state, optimizer.q_table)
        self.assertGreater(optimizer.q_table[state][2], 0) # Q-value for action 1 should be updated

        # Test reward calculation
        metrics.block_times.clear()
        metrics.block_times.extend([10, 10, 10]) # Perfect block time
        reward = optimizer.calculate_reward(metrics, target_block_time=10.0)
        self.assertAlmostEqual(reward, 1.0)

        metrics.block_times.clear()
        metrics.block_times.extend([20, 20, 20]) # Slow block time
        reward = optimizer.calculate_reward(metrics, target_block_time=10.0)
        self.assertLess(reward, 0.1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
