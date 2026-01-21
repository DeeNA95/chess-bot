import torch
import chess
import time
from src.search.mcts_cpp import MCTS
from src.core.state_encoder import StateEncoder

class MockModel(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        # Random policy, value
        logits = torch.randn(batch_size, 4096)
        values = torch.tanh(torch.randn(batch_size, 1))
        return logits, values

def test_mcts_cpp_basic():
    print("Testing C++ MCTS Basic...")
    model = MockModel()
    encoder = StateEncoder()
    mcts = MCTS(model, encoder, num_simulations=50)

    board = chess.Board()
    results = mcts.search_batch([board])

    policy, value = results[0]
    print(f"Value: {value:.4f}")
    print(f"Policy max prob: {policy.max().item():.4f}")
    assert policy.shape == (4096,)
    assert -1.0 <= value <= 1.0

def test_mcts_cpp_speed():
    print("\nTesting C++ MCTS Speed...")
    model = MockModel()
    encoder = StateEncoder()
    mcts = MCTS(model, encoder, num_simulations=100)

    boards = [chess.Board() for _ in range(32)]

    start = time.time()
    for _ in range(5): # 5 batch steps
        mcts.search_batch(boards)
    end = time.time()

    print(f"Time for 5 steps (32 games, 100 sims): {end - start:.4f}s")
    print(f"Speed: {(5 * 32 * 100) / (end - start):.0f} nodes/sec (incl python NN overhead)")

if __name__ == "__main__":
    test_mcts_cpp_basic()
    test_mcts_cpp_speed()
