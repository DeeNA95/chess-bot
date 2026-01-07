"""
Self-Play Training Loop with MCTS.

Combines:
- Self-play (agent plays both sides)
- MCTS-guided move selection
- Policy and value training from game outcomes
"""

import torch
import torch.nn.functional as F
import os
import chess
import shutil
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass

from src.core.state_encoder import StateEncoder
from src.agents.ppo_agent import PPOAgent
from src.search.mcts import MCTS
from src.utils import get_device


@dataclass
class GameSample:
    """Single training sample from a game."""
    observation: torch.Tensor  # Board state encoding
    mcts_policy: torch.Tensor  # MCTS visit distribution (4096,)
    outcome: float             # Game outcome from this player's perspective


def play_game(
    agent: PPOAgent,
    encoder: StateEncoder,
    mcts: MCTS,
    device: torch.device,
    max_moves: int = 200,
) -> List[GameSample]:
    """
    Play one self-play game using MCTS for move selection.

    Returns list of training samples (one per move).
    """
    board = chess.Board()
    samples = []
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        # Get observation
        obs = encoder.encode(board)

        # Run MCTS to get improved policy
        move, policy, value = mcts.select_move(board)

        # Store sample (outcome filled in later)
        samples.append(GameSample(
            observation=obs,
            mcts_policy=policy,
            outcome=0.0,  # Placeholder
        ))

        # Make move
        board.push(move)
        move_count += 1

    # Determine game outcome
    if board.is_checkmate():
        # Last player to move won
        winner = not board.turn  # board.turn is loser (they're in checkmate)
    else:
        winner = None  # Draw

    # Fill in outcomes from each player's perspective
    for i, sample in enumerate(samples):
        if winner is None:
            sample.outcome = 0.0  # Draw
        else:
            # Determine who moved for this sample
            mover = chess.WHITE if i % 2 == 0 else chess.BLACK
            sample.outcome = 1.0 if mover == winner else -1.0

    return samples


def train_loop(
    total_games: int = 10000,
    checkpoint_dir: str = '/checkpoints',
    num_simulations: int = 50,
    batch_size: int = 256,
    games_per_update: int = 10,
    lr: float = 1e-4,
):
    """
    Main training loop with self-play + MCTS.

    Args:
        total_games: Total self-play games to play
        checkpoint_dir: Where to save checkpoints
        num_simulations: MCTS simulations per move
        batch_size: Training batch size
        games_per_update: Games to play before each training update
        lr: Learning rate
    """
    device = get_device()
    print(f'Training on {device}')
    print(f'MCTS simulations per move: {num_simulations}')

    # Initialize components
    agent = PPOAgent(device=str(device), lr=lr)
    encoder = StateEncoder(device=str(device))

    mcts = MCTS(
        model=agent.model,
        encoder=encoder,
        device=str(device),
        num_simulations=num_simulations,
        c_puct=1.5,
        temperature=1.0,  # Exploration during training
    )

    # Optimizer for policy + value loss
    optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)

    # WandB logging
    import wandb
    wandb.init(project='chess-rl-bot', config={
        'total_games': total_games,
        'device': str(device),
        'num_simulations': num_simulations,
        'batch_size': batch_size,
        'games_per_update': games_per_update,
        'training_mode': 'mcts-self-play',
    })

    # Training loop
    all_samples: List[GameSample] = []
    games_played = 0
    update_count = 0

    print(f'Starting MCTS Self-Play Training: {total_games} games')

    while games_played < total_games:
        # Play batch of games
        for _ in range(games_per_update):
            samples = play_game(agent, encoder, mcts, device)
            all_samples.extend(samples)
            games_played += 1

            if games_played % 10 == 0:
                print(f'   > Games: {games_played}/{total_games}, Samples: {len(all_samples)}', end='\r')

        # Training update
        if len(all_samples) >= batch_size:
            metrics = train_step(agent.model, optimizer, all_samples, batch_size, device)
            update_count += 1

            # Log to WandB
            wandb.log({
                'games_played': games_played,
                'samples': len(all_samples),
                'update': update_count,
                **metrics,
            })

            print(f'\n[Game {games_played}] Update {update_count}: '
                  f'Policy Loss: {metrics["policy_loss"]:.4f}, '
                  f'Value Loss: {metrics["value_loss"]:.4f}')

            # Keep last N samples to avoid memory explosion
            max_samples = batch_size * 50
            if len(all_samples) > max_samples:
                all_samples = all_samples[-max_samples:]

        # Periodic checkpoints
        if games_played % 100 == 0:
            save_path = os.path.join(checkpoint_dir, f'mcts_game_{games_played}.pt')
            os.makedirs(checkpoint_dir, exist_ok=True)
            agent.save(save_path)
            print(f'   > Checkpoint: {save_path}')

    # Final checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_path = os.path.join(checkpoint_dir, 'mcts_final.pt')
    agent.save(save_path)
    print(f'Saved final model to {save_path}')


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    samples: List[GameSample],
    batch_size: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    One training update on a batch of samples.

    Losses:
    - Policy: Cross-entropy between network policy and MCTS policy
    - Value: MSE between network value and game outcome
    """
    # Sample a batch
    indices = torch.randperm(len(samples))[:batch_size]

    obs_batch = torch.stack([samples[i].observation for i in indices]).to(device)
    policy_batch = torch.stack([samples[i].mcts_policy for i in indices]).to(device)
    outcome_batch = torch.tensor([samples[i].outcome for i in indices], device=device)

    # Forward pass
    logits, values = model(obs_batch)
    values = values.squeeze()

    # Policy loss: cross-entropy with MCTS policy
    log_probs = F.log_softmax(logits, dim=-1)
    policy_loss = -torch.sum(policy_batch * log_probs, dim=-1).mean()

    # Value loss: MSE with game outcome
    value_loss = F.mse_loss(values, outcome_batch)

    # Combined loss
    total_loss = policy_loss + value_loss

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'total_loss': total_loss.item(),
    }


if __name__ == '__main__':
    train_loop(total_games=100, checkpoint_dir='./test_checkpoints')
