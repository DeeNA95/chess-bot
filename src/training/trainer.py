"""
Self-Play Training Loop with Batched MCTS.

Combines:
- Parallel Self-play (multiple games in flight)
- Batched MCTS move selection
- Policy and value training with AMP
"""

import torch
import torch.nn.functional as F
import os
import chess
import time
import random
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque

from src.core.state_encoder import StateEncoder
from src.agents.ppo_agent import ChessAgent
from src.search.mcts import MCTS
from src.utils import get_device

@dataclass
class GameSample:
    """Single training sample from a game."""
    observation: torch.Tensor  # Board state encoding
    mcts_policy: torch.Tensor  # MCTS visit distribution (4096,)
    outcome: float             # Game outcome from this player's perspective


class ReplayBuffer:
    """Efficient circular buffer for training samples."""
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, sample: GameSample):
        self.buffer.append(sample)

    def extend(self, samples: List[GameSample]):
        self.buffer.extend(samples)

    def sample(self, batch_size: int) -> List[GameSample]:
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)


def play_games_batch(
    mcts: MCTS,
    encoder: StateEncoder,
    num_parallel_games: int,
    device: torch.device,
    max_moves: int = 200,
) -> List[GameSample]:
    """
    Play N self-play games in parallel using Batched MCTS.
    """
    boards = [chess.Board() for _ in range(num_parallel_games)]
    game_samples = [[] for _ in range(num_parallel_games)]
    active_indices = list(range(num_parallel_games))
    finished_samples = []

    move_count = 0
    while active_indices and move_count < max_moves:
        current_boards = [boards[i] for i in active_indices]

        # 1. Batched MCTS Search
        search_results = mcts.search_batch(current_boards)

        next_active_indices = []
        for i, idx in enumerate(active_indices):
            board = boards[idx]
            policy, value = search_results[i]

            # Store sample (outcome filled later)
            obs = encoder.encode(board).cpu() # Move to CPU for storage
            game_samples[idx].append(GameSample(
                observation=obs,
                mcts_policy=policy.cpu(),
                outcome=0.0
            ))

            # 2. Select Move
            if mcts.temperature > 0:
                action_idx = int(torch.multinomial(policy, 1).item())
            else:
                action_idx = int(policy.argmax().item())

            from_sq = action_idx // 64
            to_sq = action_idx % 64
            move = chess.Move(from_sq, to_sq)

            # Handle Queen promotion
            if chess.square_rank(to_sq) in [0, 7]:
                piece = board.piece_at(from_sq)
                if piece and piece.piece_type == chess.PAWN:
                    move.promotion = chess.QUEEN

            # 3. Step Board
            board.push(move)

            # 4. Check for terminal
            if board.is_game_over():
                # Process outcome for this game
                outcome_val = 0.0
                if board.is_checkmate():
                    winner = not board.turn
                else:
                    winner = None

                # Assign outcomes to all samples in this game
                for j, sample in enumerate(game_samples[idx]):
                    mover_color = chess.WHITE if j % 2 == 0 else chess.BLACK
                    if winner is None:
                        sample.outcome = 0.0
                    else:
                        sample.outcome = 1.0 if mover_color == winner else -1.0

                finished_samples.extend(game_samples[idx])
            else:
                next_active_indices.append(idx)

        active_indices = next_active_indices
        move_count += 1

    # Force outcome for unfinished games (treat as draws or use value head)
    for idx in active_indices:
        for sample in game_samples[idx]:
            sample.outcome = 0.0 # Draw for timeout
        finished_samples.extend(game_samples[idx])

    return finished_samples


def train_loop(
    total_games: int = 10000,
    checkpoint_dir: str = '/checkpoints',
    num_simulations: int = 50,
    batch_size: int = 256,
    num_parallel_games: int = 16,
    games_per_update: int = 32,
    buffer_capacity: int = 100000,
    lr: float = 1e-4,
):
    """
    Main training loop with Batched MCTS and Mixed Precision.
    """
    device = get_device()
    print(f'Training on {device} (AMP Enabled)')

    # Initialize components
    agent = ChessAgent(device=str(device), lr=lr)
    encoder = StateEncoder(device=str(device))
    buffer = ReplayBuffer(capacity=buffer_capacity)

    # Load from latest checkpoint if exists
    start_game = 0
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('mcts_game_') and f.endswith('.pt')]
        if checkpoint_files:
            # Extract numbers and pick highest
            checkpoint_files.sort(key=lambda x: int(x.split('_')[2].split('.')[0]))
            latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[-1])
            print(f'Resuming from checkpoint: {latest_checkpoint}')
            agent.load(latest_checkpoint)
            start_game = int(checkpoint_files[-1].split('_')[2].split('.')[0])
        elif os.path.exists(os.path.join(checkpoint_dir, 'mcts_final.pt')):
            latest_checkpoint = os.path.join(checkpoint_dir, 'mcts_final.pt')
            print(f'Resuming from final checkpoint: {latest_checkpoint}')
            agent.load(latest_checkpoint)

    mcts = MCTS(
        model=agent.model,
        encoder=encoder,
        device=str(device),
        num_simulations=num_simulations,
        temperature=1.0,
    )

    optimizer = torch.optim.Adam(agent.model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    import wandb
    wandb.init(project='chess-rl-bot', config={
        'total_games': total_games,
        'num_simulations': num_simulations,
        'batch_size': batch_size,
        'num_parallel_games': num_parallel_games,
        'games_per_update': games_per_update,
        'amp': True,
    })

    games_played = start_game
    update_count = 0
    last_info_time = time.time()

    print(f'Starting Batched Self-Play: {total_games} games total')

    while games_played < total_games:
        # 1. Play Games in Parallel
        start_play = time.time()
        new_samples = play_games_batch(
            mcts, encoder, num_parallel_games, device
        )
        buffer.extend(new_samples)
        games_played += num_parallel_games
        play_time = time.time() - start_play

        # 2. Training Updates
        if len(buffer) >= batch_size:
            start_train = time.time()
            # Perform multiple updates based on games_per_update/ratio
            updates_to_run = max(1, games_per_update // num_parallel_games)

            total_metrics = {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0}
            for _ in range(updates_to_run):
                batch = buffer.sample(batch_size)
                metrics = train_step(agent.model, optimizer, scaler, batch, device)
                for k in total_metrics:
                    total_metrics[k] += metrics[k]

            for k in total_metrics:
                total_metrics[k] /= float(updates_to_run)

            update_count += 1
            train_time = time.time() - start_train

            # Logging
            if time.time() - last_info_time > 10: # Log every 10s
                games_per_sec = num_parallel_games / play_time
                print(f'[Game {games_played:5d}] Update {update_count:4d} | '
                      f'Loss: {total_metrics["total_loss"]:.4f} | '
                      f'Speed: {games_per_sec:.2f} games/s')
                last_info_time = time.time()

            wandb.log({
                'games_played': games_played,
                'buffer_size': len(buffer),
                'games_per_sec': num_parallel_games / play_time,
                **total_metrics
            })

        # Periodic Checkpoints
        if games_played % 100 < num_parallel_games and games_played > 0:
            save_path = os.path.join(checkpoint_dir, f'mcts_game_{games_played}.pt')
            os.makedirs(checkpoint_dir, exist_ok=True)
            agent.save(save_path)
            print(f'   > Checkpoint saved: {save_path}')

    # Final Save
    save_path = os.path.join(checkpoint_dir, 'mcts_final.pt')
    agent.save(save_path)
    print(f'Training complete. Final model saved to {save_path}')


def train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[Any],
    samples: List[GameSample],
    device: torch.device,
) -> Dict[str, float]:
    """
    One training step with Mixed Precision (AMP).
    """
    obs_batch = torch.stack([s.observation for s in samples]).to(device)
    policy_batch = torch.stack([s.mcts_policy for s in samples]).to(device)
    outcome_batch = torch.tensor([s.outcome for s in samples], device=device)

    # AMP Context
    dtype = torch.float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16

    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=dtype):
        logits, values = model(obs_batch)
        values = values.squeeze()

        # Policy loss: Cross-entropy
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -torch.sum(policy_batch * log_probs, dim=-1).mean()

        # Value loss: MSE
        value_loss = F.mse_loss(values, outcome_batch)

        total_loss = policy_loss + value_loss

    optimizer.zero_grad()

    if scaler:
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return {
        'policy_loss': policy_loss.item(),
        'value_loss': value_loss.item(),
        'total_loss': total_loss.item(),
    }


if __name__ == '__main__':
    train_loop(total_games=100)
