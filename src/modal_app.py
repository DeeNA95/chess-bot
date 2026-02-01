import modal
import os
import sys

app = modal.App('chess-rl-bot')

image = (
    modal.Image.debian_slim()
    .apt_install('stockfish', 'git', 'g++')
    .add_local_dir('src', remote_path='/root/src', copy=True)
    .add_local_file('config.yaml', remote_path='/root/config.yaml', copy=True)
    .add_local_file('setup.py', remote_path='/root/setup.py', copy=True)
    .add_local_file('pyproject.toml', remote_path='/root/pyproject.toml', copy=True)
    .add_local_file('README.md', remote_path='/root/README.md', copy=True)
    .run_commands("pip install uv && cd /root && uv pip install --system pybind11 && uv pip install --system .")
)

volume = modal.Volume.from_name('chess-rl-checkpoints', create_if_missing=True)


@app.function(
    image=image,
    gpu='t4',
    cpu=12.0, # High CPU for Stockfish workers
    memory=16 * 1024, # 64GB RAM
    timeout=60 * 60 * 24,  # 24 hours
    volumes={'/checkpoints': volume},
    secrets=[modal.Secret.from_name('wandb-secret')],
    retries=0
)
def train_function(config_path: str = '/root/config.yaml'):
    """Algorithm-agnostic Training - the main training entry point."""
    sys.path.append('/root')

    from src.training.trainer import train_loop

    print(f'Starting Training on Modal GPU with config: {config_path}...')

    train_loop(config_path)


@app.function(image=image, gpu='t4', cpu=2.0)
def debug_env():
    sys.path.append('/root')
    from src.core.chess_env import ChessEnv
    from src.utils import get_device

    device = get_device()
    print(f'Running Debug on Device: {device}')

    env = ChessEnv(device=str(device))
    obs, info = env.reset()
    print('Observation Shape:', obs['observation'].shape)
    print('Observation Device:', obs['observation'].device)
    print('Action Mask Device:', obs['action_mask'].device)

    import shutil
    print("shutil.which('stockfish'):", shutil.which('stockfish'))
    print('Explicit path /usr/games/stockfish exists:', os.path.exists('/usr/games/stockfish'))


@app.function(image=image, gpu='t4', cpu=4.0, retries=0)
def verify_hybrid_mcts(num_games: int = 10):
    """
    Verify Hybrid MCTS execution on Modal.
    Run a short game loop and check MCTS/Stockfish interaction.
    """
    import chess
    import torch
    from src.core.config import AppConfig
    from src.agents.ppo_agent import ChessAgent as PPOAgent
    from src.core.state_encoder import StateEncoder
    from src.search.mcts_cpp import MCTS
    from src.training.async_verifier import AsyncStockfishVerifier

    sys.path.append('/root')

    # 1. Load and Patch Config
    config = AppConfig.load('/root/config.yaml')
    config.rewards.stockfish_path = '/usr/games/stockfish'
    config.mcts.num_simulations = 50  # Reduced for faster testing

    print(f"✅ Config Loaded. Stockfish: {config.rewards.stockfish_path}")

    # 2. Init Components
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")

    encoder = StateEncoder(device=str(device))
    agent = PPOAgent(device=device)

    mcts = MCTS(
        model=agent.model,
        encoder=encoder,
        device=device,
        num_simulations=config.mcts.num_simulations,
        c_puct=config.mcts.c_puct,
        temperature=config.mcts.temperature,
    )

    # AsyncStockfishVerifier for tree search
    sv = AsyncStockfishVerifier(
        stockfish_path=config.rewards.stockfish_path,
        num_workers=8,  # Reduced workers
        depth=config.rewards.stockfish_depth,
        hash_size=config.rewards.stockfish_hash
    )

    print("🔎 AsyncStockfishVerifier Initialized")

    # 3. Test with single board first
    board = chess.Board()
    print("🏎️  Starting Single Board Search...")

    results = mcts.search_batch([board], verifier=sv)
    policy, value = results[0]

    print(f"✅ Single Board Complete. Value: {value:.4f}")
    print(f"p shape: {policy.shape}")

    # 4. Test with increasing batch sizes (including 2048 to test pool allocator)
    for batch_size in [16, 64, 128, 256, 512, 1024, 2048]:
        boards = [chess.Board() for _ in range(batch_size)]
        print(f"🔄 Testing {batch_size} boards...")
        results = mcts.search_batch(boards, verifier=sv)
        print(f"✅ {batch_size} boards complete.")

    sv.close()
    print("🎉 Hybrid MCTS Verification Passed!")


