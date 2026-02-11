
import sys
import os
import time
import torch
import chess
import numpy as np
import argparse

# Add src to pythonpath
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.agents.chess_agent import ChessAgent
from src.core.state_encoder import StateEncoder
from src.core.config import AppConfig

# Try to import C++ MCTS
try:
    from src.search.mcts_cpp import MCTS as MCTS_CPP
    HAS_CPP = True
except ImportError:
    HAS_CPP = False
    print("WARNING: C++ MCTS not found. Benchmark will fail for C++ tests.")

def benchmark_inference(agent, encoder, device, batch_size=256, num_batches=10):
    print(f"\n--- Benchmarking Raw Inference (Batch Size: {batch_size}) ---")
    board = chess.Board()
    obs = encoder.encode(board).to(device)

    # Create batch
    batch = obs.unsqueeze(0).repeat(batch_size, 1, 1, 1)

    # Warmup
    for _ in range(5):
        with torch.no_grad():
            _ = agent.model(batch)

    utils_start = time.time()
    for _ in range(num_batches):
        with torch.no_grad():
            _ = agent.model(batch)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    total_time = time.time() - utils_start

    total_samples = batch_size * num_batches
    print(f"Processed {total_samples} positions in {total_time:.4f}s")
    print(f"Throughput: {total_samples / total_time:.2f} pos/sec")

def benchmark_mcts_batch(mcts, num_games=256, sims=800):
    print(f"\n--- Benchmarking MCTS Batch Search (Games: {num_games}, Sims: {sims}) ---")
    boards = [chess.Board() for _ in range(num_games)]

    start_time = time.time()

    # Run one search step (which performs 'sims' simulations for all boards)
    # search_batch returns (policy, value) for each board
    results = mcts.search_batch(boards)

    total_time = time.time() - start_time
    print(f"Completed batch search in {total_time:.4f}s")
    print(f"Average time per game: {total_time / num_games:.4f}s")
    print(f"Effective Sims/sec: {(num_games * sims) / total_time:.2f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--sims", type=int, default=100)
    parser.add_argument("--leaves_per_sim", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Benchmarking on device: {args.device}")

    # Load default config to get model params
    config = AppConfig()

    device = torch.device(args.device)
    agent = ChessAgent(device=args.device, model_config=config.model)
    agent.model.eval()
    encoder = StateEncoder(device=args.device)

    # 1. Raw Inference Benchmark
    benchmark_inference(agent, encoder, device, batch_size=args.batch_size)

    # 2. C++ MCTS Benchmark
    if HAS_CPP:
        mcts = MCTS_CPP(
            model=agent.model,
            encoder=encoder,
            device=args.device,
            num_simulations=args.sims,
            c_puct=2.0,
            max_nodes_per_tree=200000, # Large buffer
            leaves_per_sim=args.leaves_per_sim
        )

        # We need to simulate 'leaves_per_sim' being roughly equal to batch_size
        # to saturate the GPU during MCTS expansion.
        # However, 'leaves_per_sim' in MCTS config usually controls how many leaves
        # are collected per search iteration.

        benchmark_mcts_batch(mcts, num_games=args.batch_size, sims=args.sims)

if __name__ == "__main__":
    main()
