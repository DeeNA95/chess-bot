import gymnasium as gym
import chess
import numpy as np
from src.core.state_encoder import StateEncoder
from typing import Optional, Tuple, Dict, Any, TypeAlias
import torch
from src.utils import get_device

ObsType: TypeAlias = Dict[str, Any]
InfoType: TypeAlias = Dict[str, Any]

class SelfPlayChessEnv(gym.Env):
    """
    Self-Play Chess Environment where both sides are played by the agent.

    Key differences from ChessEnv:
    - Agent plays BOTH White and Black
    - Observations are always from the current player's perspective
    - Rewards are from the perspective of the player who just moved
    - On checkmate: winner gets +1, info contains 'loser_reward' = -1
    """

    metadata = {'render_modes': ['human', 'ansi']}

    def __init__(self, render_mode=None, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.encoder = StateEncoder(device=self.device)

        self.action_space = gym.spaces.Discrete(4096)
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(low=0, high=1, shape=(116, 8, 8), dtype=np.float32),
            'action_mask': gym.spaces.Box(low=0, high=1, shape=(4096,), dtype=np.int8)
        })

        self.board = chess.Board()
        self.render_mode = render_mode

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, InfoType]:
        super().reset(seed=seed)
        self.board.reset()
        return self._get_obs(), self._get_info()

    def step(self, action: int) -> Tuple[ObsType, float, bool, bool, InfoType]:
        """
        Execute a move for the current player.

        Returns:
            obs: Observation from the NEW current player's perspective (opponent of mover)
            reward: Reward for the player who just moved
            terminated: Whether the game ended
            truncated: Whether the episode was truncated
            info: Additional info including 'mover_color' indicating who just moved
        """
        mover_color = self.board.turn  # Who is about to move

        # Decode action
        from_sq = action // 64
        to_sq = action % 64
        move = chess.Move(from_sq, to_sq)

        # Handle pawn promotion (always Queen)
        piece = self.board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            if chess.square_rank(to_sq) in [0, 7]:
                move.promotion = chess.QUEEN

        # Verify legality
        if move not in self.board.legal_moves:
            # Illegal move - severe penalty, game ends
            return self._get_obs(), -10.0, True, False, {
                'error': 'illegal_move',
                'mover_color': mover_color
            }

        # Track captured piece for material verifier
        captured_piece = self.board.piece_at(move.to_square)

        # Execute move
        self.board.push(move)

        # Determine rewards and termination
        terminated = False
        reward = 0.0  # Reward for the mover
        loser_reward = None  # Set if game ends with winner/loser

        if self.board.is_checkmate():
            terminated = True
            reward = 1.0  # Mover wins (they just delivered checkmate)
            loser_reward = -1.0  # Opponent (now current player) loses
        elif self.board.is_game_over():
            terminated = True
            reward = 0.0  # Draw - no reward for either
            loser_reward = 0.0

        info = self._get_info()
        info['mover_color'] = mover_color
        info['captured_piece'] = captured_piece.piece_type if captured_piece else None
        if loser_reward is not None:
            info['loser_reward'] = loser_reward

        # Return observation from the NEW current player's perspective
        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self) -> ObsType:
        """Get observation from current player's perspective."""
        obs_tensor = self.encoder.encode(self.board)

        mask = torch.zeros(4096, dtype=torch.bool, device=self.device)
        valid_indices = [
            move.from_square * 64 + move.to_square
            for move in self.board.legal_moves
            if not move.promotion or move.promotion == chess.QUEEN
        ]
        if valid_indices:
            mask[valid_indices] = True

        return {
            'observation': obs_tensor,
            'action_mask': mask
        }

    def _get_info(self) -> InfoType:
        return {
            'turn': self.board.turn,
            'fen': self.board.fen(),
            'move_number': self.board.fullmove_number
        }

    def render(self):
        if self.render_mode == 'ansi':
            print(self.board)

    @property
    def current_player(self) -> chess.Color:
        """Returns the color of the player to move."""
        return self.board.turn
