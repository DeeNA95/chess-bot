import gymnasium as gym
import torch
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from src.core.chess_env import ChessEnv
from concurrent.futures import ThreadPoolExecutor

class VectorChessEnv:
    """
    Vectorized Wrapper for multiple ChessEnvs with threaded execution.
    """
    def __init__(self, num_envs: int, device: str = "cpu"):
        self.num_envs = num_envs
        self.device = device
        self.envs = [ChessEnv(device=device) for _ in range(num_envs)]

        # Persistent thread pool - avoids creation overhead per step
        self._executor = ThreadPoolExecutor(max_workers=num_envs)

        # Mirror single env properties
        self.observation_space_shape = (116, 8, 8)
        self.action_mask_shape = (4096,)

    def reset(self) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, Any]]]:
        obs_list = []
        mask_list = []
        info_list = []

        # Reset can also be threaded if needed, but it's fast usually
        for env in self.envs:
            o, i = env.reset()
            obs_list.append(o['observation'])
            mask_list.append(o['action_mask'])
            info_list.append(i)

        batched_obs = {
            "observation": torch.stack(obs_list),
            "action_mask": torch.stack(mask_list)
        }

        return batched_obs, info_list

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """
        Step all environments in parallel using threads.
        actions: (B,) tensor of action indices
        """
        actions_cpu = actions.cpu().numpy()

        # Parallel execution helper
        def step_env(args):
            i, env, action = args
            o, r, t, tr, info = env.step(action)

            # Auto-reset if done
            if t or tr:
                o, info_reset = env.reset()
                info = info_reset # Update info to new episode start
            return o, r, t, tr, info

        # Execute parallel steps using persistent executor
        results = list(self._executor.map(step_env, zip(range(self.num_envs), self.envs, actions_cpu)))

        # Unpack results
        obs_list = [r[0]['observation'] for r in results]
        mask_list = [r[0]['action_mask'] for r in results]
        reward_list = [r[1] for r in results]

        # Combine term/trunc
        term_list = [float(r[2] or r[3]) for r in results]
        trunc_list = [float(r[3]) for r in results]
        info_list = [r[4] for r in results]

        batched_obs = {
            "observation": torch.stack(obs_list),
            "action_mask": torch.stack(mask_list)
        }

        rewards = torch.tensor(reward_list, device=self.device, dtype=torch.float32)
        dones = torch.tensor(term_list, device=self.device, dtype=torch.float32)

        return batched_obs, rewards, dones, torch.tensor(trunc_list), info_list

    @property
    def boards(self):
        return [env.board for env in self.envs]

    def close(self):
        """Shutdown the thread pool executor."""
        self._executor.shutdown(wait=False)
