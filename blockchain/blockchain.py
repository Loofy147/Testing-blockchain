import hashlib
import time
import json
from typing import List, Dict, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LinearRegression


@dataclass
class Transaction:
    """Structured transaction with validation"""
    sender: str
    recipient: str
    amount: float
    fee: float = 0.0
    timestamp: float = None
    signature: Optional[str] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.amount <= 0:
            raise ValueError("Transaction amount must be positive")
        if self.fee < 0:
            raise ValueError("Transaction fee cannot be negative")

    def to_dict(self):
        return {
            "sender": self.sender,
            "recipient": self.recipient,
            "amount": self.amount,
            "fee": self.fee,
            "timestamp": self.timestamp
        }

    def calculate_hash(self):
        """Hash for transaction verification"""
        tx_string = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(tx_string.encode()).hexdigest()


class Block:
    def __init__(self, index: int, timestamp: float, transactions: List[Transaction], previous_hash: str):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate block hash including all transactions"""
        tx_data = json.dumps([tx.to_dict() for tx in self.transactions], sort_keys=True)
        block_string = f"{self.index}{self.timestamp}{tx_data}{self.previous_hash}{self.nonce}"
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine_block(self, difficulty: int) -> int:
        """Mine block and return number of attempts"""
        target = "0" * difficulty
        attempts = 0

        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
            attempts += 1

            # Progress indicator for long mining
            if attempts % 10000 == 0:
                print(f"Mining... {attempts} attempts")

        return attempts


class FeePredictor:
    """Smart contract for dynamic fee prediction"""

    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False
        self.min_fee = 0.01
        self.max_fee = 10.0

    def train_model(self, historical_data: List[tuple] = None):
        """Train on historical [pending_tx_count, fee] data"""
        if historical_data is None:
            # Default training data: more pending transactions = higher fees
            X = np.array([[5], [10], [20], [30], [50], [75], [100]]).reshape(-1, 1)
            y = np.array([0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
        else:
            X = np.array([data[0] for data in historical_data]).reshape(-1, 1)
            y = np.array([data[1] for data in historical_data])

        self.model.fit(X, y)
        self.is_trained = True

    def predict_fee(self, pending_transactions_count: int) -> float:
        """Predict optimal fee based on network congestion"""
        if not self.is_trained:
            self.train_model()

        prediction = self.model.predict(np.array([[pending_transactions_count]]))[0]

        # Clamp to reasonable bounds
        return max(self.min_fee, min(self.max_fee, prediction))

    def update_with_block(self, block: Block):
        """Update model with new block data (online learning)"""
        # Could implement incremental learning here
        pass


class Blockchain:
    def __init__(self, difficulty: int = 2):
        self.chain: List[Block] = [self.create_genesis_block()]
        self.difficulty = difficulty
        self.pending_transactions: List[Transaction] = []
        self.mining_reward = 100.0
        self.fee_predictor = FeePredictor()
        self.balances: Dict[str, float] = {}

        # Initialize fee predictor
        self.fee_predictor.train_model()

    def create_genesis_block(self) -> Block:
        """Create the first block in the chain"""
        return Block(0, time.time(), [], "0")

    def get_latest_block(self) -> Block:
        return self.chain[-1]

    def get_balance(self, address: str) -> float:
        """Calculate balance for an address"""
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

    def get_recommended_fee(self) -> float:
        """Get AI-predicted fee based on current network state"""
        pending_count = len(self.pending_transactions)
        return self.fee_predictor.predict_fee(pending_count)

    def add_transaction(self, sender: str, recipient: str, amount: float, fee: float = None) -> bool:
        """Add validated transaction to pending pool"""

        # Network reward transactions bypass validation
        if sender == "network":
            tx = Transaction(sender, recipient, amount, 0)
            self.pending_transactions.append(tx)
            return True

        # Auto-calculate fee if not provided
        if fee is None:
            fee = self.get_recommended_fee()

        # Validate sender has sufficient balance
        sender_balance = self.get_balance(sender)
        total_cost = amount + fee

        if sender_balance < total_cost:
            print(f"❌ Insufficient balance. Required: {total_cost}, Available: {sender_balance}")
            return False

        try:
            tx = Transaction(sender, recipient, amount, fee)
            self.pending_transactions.append(tx)
            print(f"✅ Transaction added. Fee: {fee:.4f}")
            return True
        except ValueError as e:
            print(f"❌ Invalid transaction: {e}")
            return False

    def mine_pending_transactions(self, mining_reward_address: str) -> Block:
        """Mine pending transactions into a new block"""

        if not self.pending_transactions:
            print("⚠️  No transactions to mine")
            return None

        # Calculate total fees collected
        total_fees = sum(tx.fee for tx in self.pending_transactions)

        # Add mining reward + collected fees
        reward_amount = self.mining_reward + total_fees
        reward_tx = Transaction("network", mining_reward_address, reward_amount, 0)

        # Create new block with all pending transactions + reward
        transactions_to_mine = self.pending_transactions.copy()
        transactions_to_mine.append(reward_tx)

        new_block = Block(
            len(self.chain),
            time.time(),
            transactions_to_mine,
            self.get_latest_block().hash
        )

        print(f"⛏️  Mining block {new_block.index} with {len(transactions_to_mine)} transactions...")
        start_time = time.time()
        attempts = new_block.mine_block(self.difficulty)
        mine_time = time.time() - start_time

        self.chain.append(new_block)

        # Update balances cache
        for tx in transactions_to_mine:
            if tx.recipient in self.balances:
                self.balances[tx.recipient] += tx.amount
            if tx.sender in self.balances and tx.sender != "network":
                self.balances[tx.sender] -= (tx.amount + tx.fee)

        # Clear pending transactions
        self.pending_transactions = []

        print(f"✅ Block mined! Hash: {new_block.hash[:16]}...")
        print(f"   Attempts: {attempts}, Time: {mine_time:.2f}s, Reward: {reward_amount:.2f}")

        return new_block

    def is_chain_valid(self) -> bool:
        """Validate entire blockchain integrity"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Verify block hash
            if current_block.hash != current_block.calculate_hash():
                print(f"❌ Invalid hash at block {i}")
                return False

            # Verify chain linkage
            if current_block.previous_hash != previous_block.hash:
                print(f"❌ Invalid previous hash at block {i}")
                return False

            # Verify proof of work
            if not current_block.hash.startswith("0" * self.difficulty):
                print(f"❌ Invalid proof of work at block {i}")
                return False

        return True

    def print_chain(self):
        """Display blockchain information"""
        print("\n" + "="*60)
        print(f"BLOCKCHAIN (Difficulty: {self.difficulty})")
        print("="*60)

        for block in self.chain:
            print(f"\nBlock #{block.index}")
            print(f"  Timestamp: {time.ctime(block.timestamp)}")
            print(f"  Hash: {block.hash}")
            print(f"  Previous: {block.previous_hash}")
            print(f"  Nonce: {block.nonce}")
            print(f"  Transactions: {len(block.transactions)}")

            for tx in block.transactions:
                print(f"    • {tx.sender} → {tx.recipient}: {tx.amount:.2f} (fee: {tx.fee:.4f})")

        print("\n" + "="*60)


# Demo usage
if __name__ == "__main__":
    print("🔗 Initializing Blockchain with AI Fee Prediction...\n")

    # Create blockchain
    blockchain = Blockchain(difficulty=3)

    # Initial mining to create some funds
    print("Initial mining to create network funds...")
    blockchain.mine_pending_transactions("Alice")

    # Check recommended fee
    print(f"\n💡 Recommended fee (0 pending): {blockchain.get_recommended_fee():.4f}")

    # Add transactions with auto-calculated fees
    print("\n📝 Adding transactions...")
    blockchain.add_transaction("Alice", "Bob", 50)
    blockchain.add_transaction("Alice", "Charlie", 30)

    print(f"💡 Recommended fee (2 pending): {blockchain.get_recommended_fee():.4f}")

    blockchain.add_transaction("Alice", "Dave", 10, fee=0.5)  # Custom fee

    # Mine transactions
    blockchain.mine_pending_transactions("Bob")

    # Check balances
    print("\n💰 Balances:")
    for address in ["Alice", "Bob", "Charlie", "Dave"]:
        balance = blockchain.get_balance(address)
        print(f"   {address}: {balance:.2f}")

    # Validate chain
    print(f"\n🔐 Blockchain valid: {blockchain.is_chain_valid()}")

    # Display full chain
    blockchain.print_chain()
