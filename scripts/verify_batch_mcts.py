import torch
import time
import chess
from src.search.mcts import MCTS
from src.training.trainer import play_games_batch
from src.core.state_encoder import StateEncoder
from src.models.transformer_net import ChessTransformerNet
from src.utils import get_device

def verify_performance():
    device = get_device()
    print(f"Verifying performance on {device}")

    model = ChessTransformerNet(num_input_planes=116).to(device)
    encoder = StateEncoder(device=str(device))
    mcts = MCTS(model=model, encoder=encoder, device=str(device), num_simulations=20)

    # Warmup and benchmark inference only
    print("Benchmarking raw inference...")
    dummy_input = torch.randn(8, 116, 8, 8).to(device)
    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)

    t0 = time.time()
    for _ in range(50):
        with torch.no_grad():
            _ = model(dummy_input)
    t1 = time.time()
    print(f"Raw Inference (bs=8): {(t1-t0)/50*1000:.2f} ms/batch")

    num_games = 8
    print(f"Playing {num_games} games in parallel...")

    start_time = time.time()
    samples = play_games_batch(mcts, encoder, num_games, device, max_moves=20)
    end_time = time.time()

    duration = end_time - start_time
    print(f"Finished in {duration:.2f} seconds.")
    print(f"Collected {len(samples)} samples.")
    print(f"Throughput: {len(samples)/duration:.2f} samples/sec")

    # Basic correctness check
    assert len(samples) > 0, "No samples collected!"
    for s in samples:
        assert s.observation.shape == (116, 8, 8), f"Wrong observation shape: {s.observation.shape}"
        assert s.mcts_policy.shape == (4096,), f"Wrong policy shape: {s.mcts_policy.shape}"
        assert -1.0 <= s.outcome <= 1.0, f"Outcome out of bounds: {s.outcome}"

    print("Verification SUCCESSFUL!")

if __name__ == "__main__":
    verify_performance()
