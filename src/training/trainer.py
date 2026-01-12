import torch
import torch.nn.functional as F
import os
import chess
import time
import random
import logging
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from collections import deque

from src.core.state_encoder import StateEncoder
from src.agents.ppo_agent import ChessAgent
from src.search.mcts import MCTS
from src.rl.grpo import GRPO
from src.rl.ppo import PPO
from src.core.config import AppConfig
from src.utils import get_device

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GameSample:
    """Single training sample from a game for MCTS."""
    observation: torch.Tensor  # Board state encoding
    mcts_policy: torch.Tensor  # MCTS visit distribution (4096,)
    outcome: float             # Game outcome from this player's perspective

@dataclass
class GRPOSample:
    """Sample for GRPO training."""
    observation: torch.Tensor
    action: int
    old_log_prob: float
    reward: float
    mask: torch.Tensor

@dataclass
class PPOSample:
    """Sample for PPO training."""
    observation: torch.Tensor
    action: int
    log_prob: float
    reward: float
    value: float
    done: bool
    mask: torch.Tensor

class ReplayBuffer:
    """Efficient circular buffer for training samples."""
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, sample: Any):
        self.buffer.append(sample)

    def extend(self, samples: List[Any]):
        self.buffer.extend(samples)

    def sample(self, batch_size: int) -> List[Any]:
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self):
        return len(self.buffer)


def play_games_mcts(
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
            obs = encoder.encode(board).cpu()
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

            move = _action_to_move(board, action_idx)
            board.push(move)

            # 3. Check for terminal
            if board.is_game_over():
                _assign_outcomes(game_samples[idx], board)
                finished_samples.extend(game_samples[idx])
            else:
                next_active_indices.append(idx)

        active_indices = next_active_indices
        move_count += 1

    for idx in active_indices:
        _assign_outcomes(game_samples[idx], boards[idx], timeout=True)
        finished_samples.extend(game_samples[idx])

    return finished_samples


def play_games_grpo(
    agent: ChessAgent,
    encoder: StateEncoder,
    config: AppConfig,
    device: torch.device,
    max_moves: int = 100,
) -> List[GRPOSample]:
    """
    Play games for GRPO.
    Samples multiple outcomes (group) from same start states to compute relative advantages.
    """
    group_size = config.grpo.group_size
    num_groups = config.training.num_parallel_games // group_size
    if num_groups == 0: num_groups = 1

    all_samples = []

    for _ in range(num_groups):
        # Start from a shared initial board
        start_board = chess.Board()

        obs = encoder.encode(start_board).to(device)
        mask = encoder.get_action_mask(start_board).to(device)

        with torch.no_grad():
            logits, _ = agent.model(obs.unsqueeze(0))
            logits = logits.masked_fill(~mask.unsqueeze(0), -float('inf'))
            probs = torch.softmax(logits, dim=-1)

            # Sample group_size actions for this SAME state
            actions = torch.multinomial(probs, group_size, replacement=True).squeeze(0)
            log_probs = torch.log(probs.gather(1, actions.unsqueeze(0))).squeeze(0)

        group_rewards = []
        for i in range(group_size):
            action_idx = int(actions[i].item())
            board = start_board.copy()
            move = _action_to_move(board, action_idx)
            board.push(move)

            reward = 0.0
            if board.is_game_over():
                if board.is_checkmate():
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                with torch.no_grad():
                    _, val = agent.model(encoder.encode(board).to(device).unsqueeze(0))
                    reward = float(val.item())

            group_rewards.append(reward)
            all_samples.append(GRPOSample(
                observation=obs.cpu(),
                action=action_idx,
                old_log_prob=float(log_probs[i].item()),
                reward=reward,
                mask=mask.cpu()
            ))

    return all_samples


def play_games_ppo(
    agent: ChessAgent,
    encoder: StateEncoder,
    config: AppConfig,
    device: torch.device,
    max_moves: int = 200,
) -> List[PPOSample]:
    """
    Play games for PPO. Generates full trajectories.
    """
    num_games = config.training.num_parallel_games
    boards = [chess.Board() for _ in range(num_games)]
    samples = [[] for _ in range(num_games)]
    active_indices = list(range(num_games))
    finished_samples = []

    move_count = 0
    while active_indices and move_count < max_moves:
        current_boards = [boards[i] for i in active_indices]

        # Batch Encode
        obs_list = [encoder.encode(b) for b in current_boards]
        obs_tensor = torch.stack(obs_list).to(device)
        masks_list = [encoder.get_action_mask(b) for b in current_boards]
        masks_tensor = torch.stack(masks_list).to(device)

        with torch.no_grad():
            logits, values = agent.model(obs_tensor)
            logits = logits.masked_fill(~masks_tensor, -float('inf'))
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        next_active_indices = []
        for i, idx in enumerate(active_indices):
            board = boards[idx]
            action_idx = int(actions[i].item())

            # Store partial sample
            samples[idx].append(PPOSample(
                observation=obs_list[i].cpu(),
                action=action_idx,
                log_prob=float(log_probs[i].item()),
                reward=0.0, # Filled later
                value=float(values[i].item()),
                done=False,
                mask=masks_list[i].cpu()
            ))

            # Step
            move = _action_to_move(board, action_idx)
            board.push(move)

            if board.is_game_over():
                reward = 0.0
                if board.is_checkmate():
                    reward = 1.0 # Current player (who just moved) won?
                else:
                    reward = 0.0 # Draw

                _backfill_rewards(samples[idx], reward)
                finished_samples.extend(samples[idx])
            else:
                next_active_indices.append(idx)

        active_indices = next_active_indices
        move_count += 1

    # Handle timeouts
    for idx in active_indices:
        _backfill_rewards(samples[idx], 0.0)
        finished_samples.extend(samples[idx])

    return finished_samples

def _backfill_rewards(trajectory: List[PPOSample], final_outcome: float):
    """
    Assign rewards for self-play.
    trajectory contains moves [White, Black, White, Black...]
    final_outcome is from perspective of the LAST player who moved.
    Argument 'final_outcome' is +1 if the last mover won.
    """
    T = len(trajectory)
    for t in reversed(range(T)):
        # If I am the last mover (index T-1), I get final_outcome.
        # If I am T-2, I am opponent, I get -final_outcome.
        # This assumes zero-sum.

        # Distance from end: k = (T-1) - t
        # if k is even: same player as last mover -> +outcome
        # if k is odd: opponent -> -outcome

        k = (T - 1) - t
        sign = 1.0 if k % 2 == 0 else -1.0
        trajectory[t].reward = final_outcome * sign
        trajectory[t].done = (t == T - 1)


def _action_to_move(board: chess.Board, action_idx: int) -> chess.Move:
    from_sq = action_idx // 64
    to_sq = action_idx % 64
    move = chess.Move(from_sq, to_sq)
    if chess.square_rank(to_sq) in [0, 7]:
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            move.promotion = chess.QUEEN
    return move


def _assign_outcomes(samples: List[GameSample], board: chess.Board, timeout: bool = False):
    if timeout:
        for s in samples: s.outcome = 0.0
        return

    if board.is_checkmate():
        winner = not board.turn
    else:
        winner = None

    for j, sample in enumerate(samples):
        mover_color = chess.WHITE if j % 2 == 0 else chess.BLACK
        if winner is None:
            sample.outcome = 0.0
        else:
            sample.outcome = 1.0 if mover_color == winner else -1.0


def train_loop(config_path: str = "config.yaml"):
    """
    Main training loop supporting MCTS and GRPO.
    """
    config = AppConfig.load(config_path)
    device = get_device()
    logger.info(f'Starting training loop for algorithm: {config.algorithm.upper()} on {device}')

    agent = ChessAgent(device=str(device), lr=config.training.lr)
    encoder = StateEncoder(device=str(device))
    buffer = ReplayBuffer(capacity=config.training.buffer_capacity)

    # Load checkpoint logic
    start_game = _load_latest_checkpoint(agent, config.training.checkpoint_dir)

    # Algorithm Provider
    mcts = None
    grpo_node = None
    if config.algorithm == "mcts":
        mcts = MCTS(
            model=agent.model,
            encoder=encoder,
            device=str(device),
            num_simulations=config.mcts.num_simulations,
            c_puct=config.mcts.c_puct,
            temperature=config.mcts.temperature,
        )
    elif config.algorithm == "grpo":
        grpo_node = GRPO(config, agent.model)
    elif config.algorithm == "ppo":
        ppo_node = PPO(config, agent.model)

    optimizer = torch.optim.Adam(agent.model.parameters(), lr=config.training.lr)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    import wandb
    wandb.init(project='chess-rl-bot', config=vars(config))

    games_played = start_game
    update_count = 0
    last_info_time = time.time()

    while games_played < config.training.total_games:
        # 1. Play Games
        start_play = time.time()
        if config.algorithm == "mcts":
            new_samples = play_games_mcts(mcts, encoder, config.training.num_parallel_games, device)
        elif config.algorithm == "grpo":
            new_samples = play_games_grpo(agent, encoder, config, device)
        else: # ppo
            new_samples = play_games_ppo(agent, encoder, config, device)

        buffer.extend(new_samples)
        games_played += config.training.num_parallel_games
        play_time = time.time() - start_play

        # 2. Training Updates
        if len(buffer) >= config.training.batch_size:
            start_train = time.time()
            updates_to_run = max(1, config.training.games_per_update // config.training.num_parallel_games)

            total_metrics = {}
            for _ in range(updates_to_run):
                batch = buffer.sample(config.training.batch_size)
                if config.algorithm == "mcts":
                    metrics = train_step_mcts(agent.model, optimizer, scaler, batch, device)
                elif config.algorithm == "grpo":
                    metrics = train_step_grpo(grpo_node, optimizer, scaler, batch, device)
                else:
                    metrics = train_step_ppo(ppo_node, optimizer, scaler, batch, device)

                for k, v in metrics.items():
                    total_metrics[k] = total_metrics.get(k, 0.0) + v

            for k in total_metrics:
                total_metrics[k] /= float(updates_to_run)

            update_count += 1
            if time.time() - last_info_time > 10:
                logger.info(f'[Game {games_played:5d}] Loss: {total_metrics.get("total_loss", total_metrics.get("loss", 0.0)):.4f} | Speed: {config.training.num_parallel_games/play_time:.2f} g/s')
                last_info_time = time.time()

            wandb.log({'games_played': games_played, **total_metrics})

        # Checkpoint
        if games_played % 100 < config.training.num_parallel_games:
            save_path = os.path.join(config.training.checkpoint_dir, f'{config.algorithm}_game_{games_played}.pt')
            agent.save(save_path)
            logger.info(f'Checkpoint saved: {save_path}')

    agent.save(os.path.join(config.training.checkpoint_dir, f'{config.algorithm}_final.pt'))


def _load_latest_checkpoint(agent, checkpoint_dir) -> int:
    if not os.path.exists(checkpoint_dir): return 0
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    if not files: return 0

    try:
        files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest = os.path.join(checkpoint_dir, files[-1])
        agent.load(latest)
        logger.info(f'Resumed from {latest}')
        return int(files[-1].split('_')[-1].split('.')[0])
    except:
        return 0


def train_step_mcts(model, optimizer, scaler, samples: List[GameSample], device) -> Dict[str, float]:
    obs_batch = torch.stack([s.observation for s in samples]).to(device)
    policy_batch = torch.stack([s.mcts_policy for s in samples]).to(device)
    outcome_batch = torch.tensor([s.outcome for s in samples], device=device)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=dtype):
        logits, values = model(obs_batch)
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -torch.sum(policy_batch * log_probs, dim=-1).mean()
        value_loss = F.mse_loss(values.squeeze(), outcome_batch)
        total_loss = policy_loss + value_loss

    optimizer.zero_grad()
    if scaler:
        scaler.scale(total_loss).backward()
        scaler.step(optimizer); scaler.update()
    else:
        total_loss.backward(); optimizer.step()

    return {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'total_loss': total_loss.item()}


def train_step_grpo(grpo, optimizer, scaler, samples: List[GRPOSample], device) -> Dict[str, float]:
    obs_batch = torch.stack([s.observation for s in samples]).to(device)
    actions = torch.tensor([s.action for s in samples], device=device)
    old_log_probs = torch.tensor([s.old_log_prob for s in samples], device=device)
    rewards = torch.tensor([s.reward for s in samples], device=device)
    masks = torch.stack([s.mask for s in samples]).to(device)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=dtype):
        metrics = grpo.compute_loss(obs_batch, actions, old_log_probs, rewards, masks)
        loss = metrics['loss']

    optimizer.zero_grad()
    if scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
    else:
        loss.backward(); optimizer.step()

    if random.random() < 0.1:
        grpo.update_ref_model()

    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}


def train_step_ppo(ppo_node, optimizer, scaler, samples: List[PPOSample], device) -> Dict[str, float]:
    obs_batch = torch.stack([s.observation for s in samples]).to(device)
    actions = torch.tensor([s.action for s in samples], device=device)
    old_log_probs = torch.tensor([s.log_prob for s in samples], device=device)
    masks = torch.stack([s.mask for s in samples]).to(device)

    rewards = torch.tensor([s.reward for s in samples], device=device)
    values = torch.tensor([s.value for s in samples], device=device)

    returns = rewards # MC return
    advantages = returns - values

    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda'), dtype=dtype):
        metrics = ppo_node.compute_loss(
             obs_batch, actions, old_log_probs, returns, advantages, masks
        )
    loss = metrics["loss"]

    optimizer.zero_grad()
    if scaler:
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
    else:
        loss.backward(); optimizer.step()

    return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()}


if __name__ == '__main__':
    train_loop()
