import torch
import numpy as np
from typing import List, Tuple, Dict, Any
from src.core.self_play_env import SelfPlayChessEnv
from concurrent.futures import ThreadPoolExecutor

class VectorSelfPlayEnv:
    """
    Vectorized Self-Play Chess Environment.

    Runs N parallel self-play games where both sides are played by the agent.
    Each step returns experiences from the player who just moved.
    """

    def __init__(self, num_envs: int, device: str = 'cpu'):
        self.num_envs = num_envs
        self.device = device
        self.envs = [SelfPlayChessEnv(device=device) for _ in range(num_envs)]

        # Persistent thread pool
        self._executor = ThreadPoolExecutor(max_workers=num_envs)

        # Mirror single env properties
        self.observation_space_shape = (116, 8, 8)
        self.action_mask_shape = (4096,)

    def reset(self) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        """Reset all environments and return initial observations."""
        obs_list = []
        mask_list = []
        info_list = []

        for env in self.envs:
            o, i = env.reset()
            obs_list.append(o['observation'])
            mask_list.append(o['action_mask'])
            info_list.append(i)

        batched_obs = {
            'observation': torch.stack(obs_list),
            'action_mask': torch.stack(mask_list)
        }
        return batched_obs, info_list

    def step(self, actions: torch.Tensor) -> Tuple[
        Dict[str, torch.Tensor],  # next_obs (from new current player's perspective)
        torch.Tensor,             # rewards (for player who just moved)
        torch.Tensor,             # dones
        torch.Tensor,             # truncated
        List[Dict[str, Any]]      # infos
    ]:
        """
        Step all environments with given actions.

        Returns observations from the NEW current player's perspective,
        and rewards for the player who just moved.
        """
        actions_cpu = actions.cpu().numpy()

        def step_env(args):
            i, env, action = args
            o, r, t, tr, info = env.step(action)

            # Auto-reset on termination
            if t or tr:
                o, reset_info = env.reset()
                # Preserve terminal info but update obs
                info['terminal_observation'] = True
                info.update(reset_info)

            return o, r, t, tr, info

        # Parallel execution
        results = list(self._executor.map(
            step_env,
            zip(range(self.num_envs), self.envs, actions_cpu)
        ))

        # Unpack results
        obs_list = [r[0]['observation'] for r in results]
        mask_list = [r[0]['action_mask'] for r in results]
        reward_list = [r[1] for r in results]
        term_list = [float(r[2] or r[3]) for r in results]
        trunc_list = [float(r[3]) for r in results]
        info_list = [r[4] for r in results]

        batched_obs = {
            'observation': torch.stack(obs_list),
            'action_mask': torch.stack(mask_list)
        }

        rewards = torch.tensor(reward_list, device=self.device, dtype=torch.float32)
        dones = torch.tensor(term_list, device=self.device, dtype=torch.float32)
        truncated = torch.tensor(trunc_list, device=self.device, dtype=torch.float32)

        return batched_obs, rewards, dones, truncated, info_list

    @property
    def boards(self) -> List:
        """Return all board states for verifier access."""
        return [env.board for env in self.envs]

    @property
    def current_players(self) -> List:
        """Return current player (to move) for each environment."""
        return [env.current_player for env in self.envs]

    def close(self):
        """Shutdown thread pool executor."""
        self._executor.shutdown(wait=False)
