import hashlib
import time
import json
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from collections import deque
import random


@dataclass
class Transaction:
    sender: str
    recipient: str
    amount: float
    fee: float
    timestamp: float
    priority_score: float = 0.0  # AI-assigned priority

    def to_dict(self):
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "fee": self.fee,
            "timestamp": self.timestamp,
            "priority_score": self.priority_score
        }

    def calculate_hash(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()


class NetworkMetrics:
    """Real-time network health monitoring"""

    def __init__(self):
        self.tx_history = deque(maxlen=100)
        self.block_times = deque(maxlen=50)
        self.fee_history = deque(maxlen=100)

    def record_transaction(self, tx: Transaction):
        self.tx_history.append({
            'timestamp': tx.timestamp,
            'fee': tx.fee,
            'amount': tx.amount
        })
        self.fee_history.append(tx.fee)

    def record_block(self, block_time: float):
        self.block_times.append(block_time)

    def get_congestion_score(self) -> float:
        """0 = empty, 1 = maximum congestion"""
        if len(self.tx_history) < 10:
            return 0.0

        recent_tx_rate = len(self.tx_history) / 100.0
        return min(1.0, recent_tx_rate)

    def get_velocity_trend(self) -> float:
        """Positive = accelerating, Negative = decelerating, based on timestamps"""
        if len(self.tx_history) < 20:
            return 0.0

        sorted_history = sorted(self.tx_history, key=lambda x: x['timestamp'])

        recent_tx = sorted_history[-10:]
        older_tx = sorted_history[-20:-10]

        if not recent_tx or not older_tx:
            return 0.0

        recent_duration = recent_tx[-1]['timestamp'] - recent_tx[0]['timestamp']
        older_duration = older_tx[-1]['timestamp'] - older_tx[0]['timestamp']

        recent_rate = len(recent_tx) / max(recent_duration, 1.0)  # tx/sec
        older_rate = len(older_tx) / max(older_duration, 1.0)    # tx/sec

        return recent_rate - older_rate

    def get_average_block_time(self) -> float:
        if not self.block_times:
            return 10.0
        return np.mean(self.block_times)


from ai.fee_predictor_lstm import FeePredictorLSTM
from ai.protocol_optimizer_rl import ProtocolOptimizerRL

class Block:
    def __init__(self, index: int, timestamp: float, transactions: List[Transaction], previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

        # AI metrics
        self.total_fees = sum(tx.fee for tx in transactions)
        self.avg_priority = np.mean([tx.priority_score for tx in transactions]) if transactions else 0

    def calculate_hash(self) -> str:
        tx_data = json.dumps([tx.to_dict() for tx in self.transactions], sort_keys=True)
        block_string = f"{self.index}{self.timestamp}{tx_data}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty: int) -> int:
        target = "0" * difficulty
        attempts = 0

        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
            attempts += 1

        return attempts

class IntelligentFeeMarket:
    """
    AI-driven dynamic fee market that learns optimal pricing
    Uses a predictive LSTM model for fee forecasting.
    """

    def __init__(self):
        # Market state
        self.base_fee = 0.01
        self.learning_rate = 0.05

        # AI Components
        self.lstm_predictor = FeePredictorLSTM(n_features=2, n_steps=10)

        # Historical performance for learning and training
        self.fee_data_history = deque(maxlen=200)

    def calculate_dynamic_fee(self,
                            pending_count: int,
                            network_metrics: NetworkMetrics,
                            user_urgency: float = 0.5) -> Dict:
        """
        Multi-factor fee calculation with explainability
        """
        # 1. Predict future base fee using the LSTM model, if available
        predicted_base_fee = 0.0
        if self.lstm_predictor:
            predicted_base_fee = self.lstm_predictor.predict(list(self.fee_data_history))

        # 2. Calculate current condition multiplier (as before, but simplified)
        congestion = network_metrics.get_congestion_score()
        pending_pressure = pending_count / 50.0

        # The multiplier now reacts to immediate conditions on top of the predicted base
        current_conditions_multiplier = 1.0 + congestion + pending_pressure

        # 3. Apply user urgency
        urgency_adjustment = 1.0 + (user_urgency - 0.5)

        # 4. Calculate final fee
        final_fee = (self.base_fee + predicted_base_fee) * current_conditions_multiplier * urgency_adjustment

        # Ensure a minimum fee
        final_fee = max(0.001, final_fee)

        return {
            'recommended_fee': round(final_fee, 4),
            'confidence': 0.85 if self.lstm_predictor and self.lstm_predictor.is_trained else 0.5, # Simplified confidence
            'explanation': {
                'predicted_base_fee': round(predicted_base_fee, 4),
                'current_conditions_multiplier': round(current_conditions_multiplier, 2),
                'urgency_adjustment': round(urgency_adjustment, 2)
            },
            'price_tiers': {
                'economy': round(final_fee * 0.7, 4),
                'standard': round(final_fee, 4),
                'priority': round(final_fee * 1.5, 4),
            }
        }

    def learn_from_block(self, block: Block, network_metrics: NetworkMetrics):
        """
        Updates historical data and retrains the LSTM model with the latest block info.
        """
        if not block.transactions:
            return

        # Get the average fee and congestion from the last block
        avg_fee_in_block = np.mean([tx.fee for tx in block.transactions if tx.sender != 'network'])
        congestion_at_block = network_metrics.get_congestion_score()

        # Add to our historical data for training
        self.fee_data_history.append((congestion_at_block, avg_fee_in_block))

        # Periodically retrain the LSTM model with the new data, if it exists
        if self.lstm_predictor and len(self.fee_data_history) > self.lstm_predictor.n_steps and block.index % 5 == 0:
            self.lstm_predictor.train(list(self.fee_data_history))


class IntelligentMempool:
    """AI-powered transaction pool with smart ordering"""

    def __init__(self, fee_market: IntelligentFeeMarket):
        self.pending: List[Transaction] = []
        self.fee_market = fee_market

    def add_transaction(self, tx: Transaction) -> Dict:
        """Add transaction with AI priority scoring"""

        # Calculate priority score based on multiple factors
        base_priority = tx.fee / max(tx.amount, 0.01)

        # Time factor: older transactions get priority boost
        age_seconds = time.time() - tx.timestamp
        age_priority = min(1.0, age_seconds / 3600)  # Max boost after 1 hour

        # Sender reputation (simplified - could be on-chain history)
        sender_priority = 0.5  # Placeholder

        tx.priority_score = (
            base_priority * 0.5 +
            age_priority * 0.3 +
            sender_priority * 0.2
        )

        self.pending.append(tx)

        return {
            'position': self._estimate_position(tx),
            'estimated_wait': self._estimate_wait_time(tx),
            'priority_score': round(tx.priority_score, 3)
        }

    def _estimate_position(self, tx: Transaction) -> int:
        """Where in the queue is this transaction?"""
        higher_priority = sum(1 for t in self.pending if t.priority_score > tx.priority_score)
        return higher_priority + 1

    def _estimate_wait_time(self, tx: Transaction) -> float:
        """Estimated seconds until inclusion"""
        position = self._estimate_position(tx)
        avg_block_time = 10.0
        tx_per_block = 10

        blocks_to_wait = position / tx_per_block
        return blocks_to_wait * avg_block_time

    def get_optimal_transactions(self, max_count: int = 10) -> List[Transaction]:
        """AI-selected optimal transaction set for next block"""

        # Sort by priority score (descending)
        sorted_tx = sorted(self.pending, key=lambda t: t.priority_score, reverse=True)

        # Optimize for: total fees, diversity, fairness
        selected = []
        total_fees = 0
        sender_counts = {}

        for tx in sorted_tx:
            if len(selected) >= max_count:
                break

            # Prevent single sender dominance
            sender_count = sender_counts.get(tx.sender, 0)
            if sender_count >= 3:
                continue

            selected.append(tx)
            total_fees += tx.fee
            sender_counts[tx.sender] = sender_count + 1

        return selected

    def remove_transactions(self, transactions: List[Transaction]):
        """Remove mined transactions from pool"""
        tx_hashes = {tx.calculate_hash() for tx in transactions}
        self.pending = [tx for tx in self.pending if tx.calculate_hash() not in tx_hashes]


class IntelligentBlockchain:
    """
    Blockchain with AI-driven economic protocol
    """

    def __init__(self, difficulty: int = 2):
        self.chain: List[Block] = [self._create_genesis_block()]
        self.difficulty = difficulty
        self.mining_reward = 100.0

        # AI components
        self.metrics = NetworkMetrics()
        self.fee_market = IntelligentFeeMarket()
        self.mempool = IntelligentMempool(self.fee_market)
        self.protocol_optimizer = ProtocolOptimizerRL(actions=[-1, 0, 1]) # Decrease, Keep, Increase difficulty

        # State
        self.balances: Dict[str, float] = {}
        self.transaction_outcomes: Dict[str, Dict] = {}
        self.last_state = None
        self.last_action = None

    def _create_genesis_block(self) -> Block:
        return Block(0, time.time(), [], "0")

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def get_balance(self, address: str) -> float:
        if address in self.balances:
            return self.balances[address]

        balance = 0.0
        for block in self.chain:
            for tx in block.transactions:
                if tx.recipient == address:
                    balance += tx.amount
                if tx.sender == address:
                    balance -= (tx.amount + tx.fee)

        self.balances[address] = balance
        return balance

    def get_fee_recommendation(self, urgency: float = 0.5) -> Dict:
        """
        Get AI-powered fee recommendation with full transparency

        urgency: 0.0 (no hurry) to 1.0 (maximum urgency)
        """
        return self.fee_market.calculate_dynamic_fee(
            pending_count=len(self.mempool.pending),
            network_metrics=self.metrics,
            user_urgency=urgency
        )

    def add_transaction(self, sender: str, recipient: str, amount: float,
                       fee: float = None, urgency: float = 0.5) -> Dict:
        """
        Add transaction with intelligent fee suggestion
        """

        # Auto-calculate optimal fee if not provided
        if fee is None:
            fee_data = self.get_fee_recommendation(urgency)
            fee = fee_data['recommended_fee']
            print(f"💡 AI recommended fee: {fee:.4f} (confidence: {fee_data['confidence']})")
            print(f"   Factors: {fee_data['explanation']}")

        # Validate balance (skip for network rewards)
        if sender != "network":
            balance = self.get_balance(sender)
            if balance < amount + fee:
                return {
                    'success': False,
                    'error': f'Insufficient balance: {balance:.2f} < {amount + fee:.2f}'
                }

        # Create and add transaction
        tx = Transaction(sender, recipient, amount, fee, time.time())
        mempool_info = self.mempool.add_transaction(tx)
        self.metrics.record_transaction(tx)

        return {
            'success': True,
            'transaction': tx,
            'mempool_info': mempool_info,
            'fee_paid': fee
        }

    def mine_block(self, miner_address: str) -> Block:
        """Mine next block with AI-optimized transaction selection"""

        if not self.mempool.pending:
            print("⚠️  No transactions in mempool")
            return None

        # AI selects optimal transaction set
        selected_tx = self.mempool.get_optimal_transactions(max_count=10)

        if not selected_tx:
            print("⚠️  No eligible transactions")
            return None

        # Calculate miner reward
        total_fees = sum(tx.fee for tx in selected_tx)
        reward_tx = Transaction("network", miner_address, self.mining_reward + total_fees, 0, time.time())

        # Create block
        block_tx = selected_tx + [reward_tx]
        new_block = Block(len(self.chain), time.time(), block_tx, self.get_latest_block().hash)

        # Mine
        print(f"⛏️  Mining block {new_block.index} with {len(selected_tx)} transactions...")
        start_time = time.time()
        attempts = new_block.mine_block(self.difficulty)
        mine_time = time.time() - start_time

        self.chain.append(new_block)
        self.metrics.record_block(mine_time)

        # AI Learning Step: Update the fee market model with the new block data
        self.fee_market.learn_from_block(new_block, self.metrics)

        # --- Reinforcement Learning Step for Protocol Optimization (if enabled) ---
        if self.protocol_optimizer:
            # 1. Get the new state from the network metrics
            next_state = self.protocol_optimizer.get_state(self.metrics)

            # 2. Calculate the reward based on the last block's performance
            reward = self.protocol_optimizer.calculate_reward(self.metrics)

            # 3. Update the Q-table with the outcome of the last action
            if self.last_state is not None and self.last_action is not None:
                self.protocol_optimizer.update_q_table(self.last_state, self.last_action, reward, next_state)

            # 4. Choose the next action (adjust difficulty)
            action = self.protocol_optimizer.choose_action(next_state)
            self.difficulty = max(1, self.difficulty + action) # Ensure difficulty is at least 1

            # 5. Store the current state and action for the next learning cycle
            self.last_state = next_state
            self.last_action = action

            # Decay exploration rate
            self.protocol_optimizer.decay_exploration()

        # Update state after successful mining
        self.mempool.remove_transactions(selected_tx)

        # Update balances cache
        for tx in block_tx:
            if tx.sender != "network":
                if tx.sender in self.balances:
                    self.balances[tx.sender] -= (tx.amount + tx.fee)
                else:
                    # This case should ideally not happen if balance is checked before adding tx
                    self.balances[tx.sender] = self.get_balance(tx.sender)

            if tx.recipient in self.balances:
                self.balances[tx.recipient] += tx.amount
            else:
                self.balances[tx.recipient] = self.get_balance(tx.recipient)

        print(f"✅ Block mined! Total fees: {total_fees:.4f}, Time: {mine_time:.2f}s")
        print(f"   Average priority: {new_block.avg_priority:.3f}")

        return new_block

    def print_ai_insights(self):
        """Display AI learning and predictions"""
        print("\n" + "="*60)
        print("🧠 AI INSIGHTS")
        print("="*60)

        print(f"\nMarket Learning:")
        print(f"  Base Fee: {self.fee_market.base_fee:.4f}")
        print(f"  Learning Rate: {self.fee_market.learning_rate}")
        print(f"  LSTM Model Trained: {'Yes' if self.fee_market.lstm_predictor.is_trained else 'No'}")

        print(f"\nNetwork State:")
        print(f"  Congestion: {self.metrics.get_congestion_score():.2%}")
        print(f"  Velocity Trend: {self.metrics.get_velocity_trend():+.3f}")
        print(f"  Avg Block Time: {self.metrics.get_average_block_time():.2f}s")

        print(f"\nMempool Status:")
        print(f"  Pending Transactions: {len(self.mempool.pending)}")
        if self.mempool.pending:
            priorities = [tx.priority_score for tx in self.mempool.pending]
            print(f"  Priority Range: {min(priorities):.3f} - {max(priorities):.3f}")

        print(f"\nProtocol Optimizer (RL):")
        print(f"  Current Difficulty: {self.difficulty}")
        print(f"  Exploration Rate (epsilon): {self.protocol_optimizer.epsilon:.3f}")
        if self.protocol_optimizer.q_table:
            print(f"  Learned States: {len(self.protocol_optimizer.q_table)}")


# Demo: Intelligent Blockchain in Action
if __name__ == "__main__":
    print("🧠 Initializing Intelligent Blockchain...\n")

    blockchain = IntelligentBlockchain(difficulty=2)

    # Bootstrap with initial funds
    blockchain.mine_block("Alice")

    print("\n" + "="*60)
    print("SCENARIO: Network Under Variable Load")
    print("="*60)

    # Simulate varying network conditions
    scenarios = [
        ("Low Load", 3, 0.3),
        ("Medium Load", 7, 0.5),
        ("High Load", 15, 0.8),
        ("Peak Load", 25, 1.0)
    ]

    for scenario_name, tx_count, urgency in scenarios:
        print(f"\n📊 {scenario_name}: Adding {tx_count} transactions (urgency={urgency})")

        for i in range(tx_count):
            result = blockchain.add_transaction(
                "Alice",
                random.choice(["Bob", "Charlie", "Dave", "Eve"]),
                random.uniform(1, 10),
                urgency=urgency
            )

            if result['success']:
                info = result['mempool_info']
                print(f"  ✓ TX queued - Position: {info['position']}, "
                      f"Wait: ~{info['estimated_wait']:.1f}s, "
                      f"Priority: {info['priority_score']:.3f}")

        # Show current fee recommendation
        rec = blockchain.get_fee_recommendation(urgency)
        print(f"\n  💰 Current Fee Tiers:")
        for tier, fee in rec['price_tiers'].items():
            print(f"     {tier.title()}: {fee:.4f}")

        # Mine block
        time.sleep(0.1)  # Simulate time passage
        blockchain.mine_block("Miner1")

    # Show AI learning outcomes
    blockchain.print_ai_insights()

    print("\n" + "="*60)
    print("🎯 Demonstration Complete")
    print("="*60)
