import modal
import os
import sys

app = modal.App('chess-rl-bot')

image = (
    modal.Image.debian_slim()
    .apt_install('stockfish', 'git')
    .pip_install(
        'torch',
        'numpy',
        'gymnasium',
        'chess',
        'wandb',
        'verifiers',
        'fastapi',
        'uvicorn'
    )
    .add_local_dir('src', remote_path='/root/src')
)

volume = modal.Volume.from_name('chess-rl-checkpoints', create_if_missing=True)


@app.function(
    image=image,
    gpu='t4',
    cpu=2.0,
    timeout=60 * 60 * 24,  # 24 hours
    volumes={'/checkpoints': volume},
    secrets=[modal.Secret.from_name('wandb-secret')]
)
def train_function(
    total_games: int = 10,
    num_simulations: int = 50,
    num_parallel_games: int = 32,
    games_per_update: int = 64,
):
    """MCTS Self-Play Training - the main training entry point."""
    sys.path.append('/root')

    from src.training.trainer import train_loop

    print('Starting MCTS Self-Play Training on Modal GPU...')
    train_loop(
        total_games=total_games,
        checkpoint_dir='/checkpoints',
        num_simulations=num_simulations,
        num_parallel_games=num_parallel_games,
        games_per_update=games_per_update,
    )


@app.function(image=image, gpu='t4', cpu=8.0)
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
