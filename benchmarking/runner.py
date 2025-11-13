import sys
import pandas as pd
import time

sys.path.insert(0, '.')

from simulation.runner import SimulationRunner

def run_benchmark(scenario: str, duration: int):
    """
    Runs a side-by-side comparison of the intelligent vs. baseline blockchain.
    """
    print("="*80)
    print(f"🚀 STARTING BENCHMARK: Scenario='{scenario}', Duration={duration} steps")
    print("="*80)

    # --- Run 1: Intelligent Blockchain (AI Enabled) ---
    print("\n\n--- 🧠 RUNNING: Intelligent Blockchain ---")
    intelligent_runner = SimulationRunner(scenario, duration)
    intelligent_runner.run()
    intelligent_results = pd.DataFrame(intelligent_runner.history)

    # --- Run 2: Baseline Blockchain (AI Disabled) ---
    print("\n\n---  पत्थर RUNNING: Baseline Blockchain (Static) ---")
    baseline_runner = SimulationRunner(scenario, duration)
    # Disable AI features for the baseline run
    baseline_runner.blockchain.protocol_optimizer = None # Disable RL optimizer
    baseline_runner.blockchain.fee_market.lstm_predictor = None # Disable LSTM
    baseline_runner.blockchain.difficulty = 3 # Set a static difficulty

    baseline_runner.run()
    baseline_results = pd.DataFrame(baseline_runner.history)

    # --- Compare Results ---
    print("\n\n" + "="*80)
    print("📊 BENCHMARK RESULTS")
    print("="*80)

    compare_metrics(intelligent_results, baseline_results)

def compare_metrics(intelligent: pd.DataFrame, baseline: pd.DataFrame):
    """Calculates and prints a comparison of key performance indicators."""

    if intelligent.empty or baseline.empty:
        print("One of the simulations produced no data. Cannot compare.")
        return

    # --- Throughput ---
    intelligent_throughput = intelligent['tx_in_block'].sum() / intelligent['timestamp'].iloc[-1] - intelligent['timestamp'].iloc[0]
    baseline_throughput = baseline['tx_in_block'].sum() / baseline['timestamp'].iloc[-1] - baseline['timestamp'].iloc[0]

    # --- Fee Analysis ---
    intelligent_avg_fee = intelligent['total_fees'].sum() / intelligent['tx_in_block'].sum()
    baseline_avg_fee = baseline['total_fees'].sum() / baseline['tx_in_block'].sum()

    # --- Block Time ---
    intelligent_block_time = (intelligent['timestamp'].iloc[-1] - intelligent['timestamp'].iloc[0]) / len(intelligent)
    baseline_block_time = (baseline['timestamp'].iloc[-1] - baseline['timestamp'].iloc[0]) / len(baseline)

    print("\n--- Key Performance Indicators ---")
    print(f"| Metric                    | Intelligent AI      | Baseline Static     | Change              |")
    print(f"|---------------------------|---------------------|---------------------|---------------------|")
    print(f"| Transaction Throughput    | {intelligent_throughput:.2f} tx/s        | {baseline_throughput:.2f} tx/s        | {((intelligent_throughput - baseline_throughput) / baseline_throughput):.2%}        |")
    print(f"| Average Transaction Fee   | {intelligent_avg_fee:.4f}           | {baseline_avg_fee:.4f}           | {((intelligent_avg_fee - baseline_avg_fee) / baseline_avg_fee):.2%}           |")
    print(f"| Average Block Time        | {intelligent_block_time:.2f} s           | {baseline_block_time:.2f} s           | {((intelligent_block_time - baseline_block_time) / baseline_block_time):.2%}           |")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python benchmarking/runner.py <scenario> <duration>")
        sys.exit(1)

    scenario = sys.argv[1]
    duration = int(sys.argv[2])

    run_benchmark(scenario, duration)
