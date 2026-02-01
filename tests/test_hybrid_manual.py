import sys
import os
import torch
import chess

# Ensure root is in path
sys.path.append(os.getcwd())

# Mock Verifier
class MockVerifier:
    def verify_batch(self, fens):
        print(f"🔎 Stockfish Verifier called for {len(fens)} positions")
        # Return dummy values
        return [0.1] * len(fens)

try:
    from src.search.mcts_cpp import MCTS
    print("✅ Successfully imported MCTS from mcts_cpp")
except ImportError as e:
    print(f"❌ Failed to import MCTS: {e}")
    sys.exit(1)

def test_hybrid_mcts():
    # Setup
    mock_model = lambda x: (torch.randn(len(x), 4096), torch.randn(len(x), 1))
    # StateEncoder mock logic (just identity or dummy)
    # Actually mcts_cpp.py uses mcts_cpp.encode_batch which is C++, so we just need a dummy Python wrapper if needed
    # But MCTS class interacts with C++ MCTS object directly.

    # We need to mock 'encoder' object if MCTS constructor uses it?
    # MCTS init: self.encoder = encoder
    # It passes encoder to nothing in init.

    device = 'cpu'

    mcts = MCTS(
        model=mock_model,
        encoder=None, # Not used in C++ logic directly, only in python wrapper
        device=device,
        num_simulations=50, # run a few sims
        c_puct=1.5
    )

    board = chess.Board()

    print("🚀 Starting Search Batch...")
    verifier = MockVerifier()

    # Run search
    results = mcts.search_batch([board], verifier=verifier)

    print(f"✅ Search finished. Result policy shape: {results[0][0].shape}, value: {results[0][1]}")

    # Verify bindings availability
    # Check if update_value is present (we can't easily check internal C++ calls from here without inspecting C++ object)
    # But if verify_batch was called (printed), then logic is working.

if __name__ == "__main__":
    test_hybrid_mcts()
