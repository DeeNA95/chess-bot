import gymnasium as gym
import chess
import numpy as np
from src.core.state_encoder import StateEncoder
from src.core.action_encoding import action_to_move, get_action_mask, ACTION_SPACE_SIZE
from typing import Optional, Tuple, Dict, Any, TypeAlias
import torch
from src.utils import get_device


ObsType: TypeAlias = Dict[str, Any]
InfoType: TypeAlias = Dict[str, Any]

class ChessEnv(gym.Env):
    """
    Chess Environment compatible with Gymnasium.

    Observation Space:
        Dict:
            'observation': (20, 8, 8) float32 tensor (Canonical Board)
            'action_mask': (4672,) bool array (Legal moves mask)

    Action Space:
        Discrete(4672) representing AlphaZero-encoded action index.
        Simplification: Always promotes to Queen.
    """

    metadata = {"render_modes": ["human", "ansi"]}


    def __init__(self, render_mode=None, device=None):
        super().__init__()
        self.device = device if device is not None else get_device()
        self.encoder = StateEncoder(device=self.device)

        # 73 move types x 64 from-squares = 4672 AlphaZero action space
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)

        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=0, high=1, shape=(116, 8, 8), dtype=np.float32),
            "action_mask": gym.spaces.Box(low=0, high=1, shape=(ACTION_SPACE_SIZE,), dtype=np.int8)
        })

        self.board = chess.Board()
        self.render_mode = render_mode

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, InfoType]:
        super().reset(seed=seed)
        self.board.reset()

        return self._get_obs(), self._get_info()

    def step(self, action):
        move = action_to_move(action, self.board)

        # Verify legality
        if move not in self.board.legal_moves:
            # Illegal move attempted. This should be masked out by the agent.
            # But if it happens, we punish severeley and end game?
            # Or just ignore and return error?
            # Standard RL practice: Agent should learn from mask. If it bypasses mask, huge penalty.
            return self._get_obs(), -10.0, True, False, {"error": "illegal_move"}

        # Check for capture
        captured_piece = self.board.piece_at(move.to_square)
        info = {"captured_piece": captured_piece.piece_type if captured_piece else None}

        # Execute Move
        self.board.push(move)

        # Check termination
        terminated = False
        truncated = False # Stalemate, etc?
        reward = 0.0

        if self.board.is_checkmate():
            terminated = True
            reward = 1.0 # Win
        elif self.board.is_game_over(): # Draw, Stalemate, etc
            terminated = True
            reward = 0.0 #TODO

        full_info = self._get_info()
        full_info.update(info)
        return self._get_obs(), reward, terminated, truncated, full_info

    def _get_obs(self):
        # Get canonical encoding (Torch Tensor)
        obs_tensor = self.encoder.encode(self.board)

        # Get action mask (Torch Tensor) from encoder
        mask = get_action_mask(self.board, device=self.device)

        return {
            "observation": obs_tensor,
            "action_mask": mask
        }

    def _get_info(self):
        return {"turn": self.board.turn, "fen": self.board.fen()}

    def render(self):
        if self.render_mode == "ansi":
            print(self.board)
