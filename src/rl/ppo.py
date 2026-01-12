import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from src.core.config import AppConfig

class PPO:
    """
    Proximal Policy Optimization (PPO).

    Standard implementation with:
    - Clipped Surrogate Objective
    - Value Function Loss (MSE)
    - Entropy Bonus
    - Generalized Advantage Estimation (GAE) (typically computed during rollout, but here we provide helpers)
    """
    def __init__(self, config: AppConfig, model: nn.Module):
        self.config = config.ppo
        self.device = config.training.device or 'cpu'
        self.model = model

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO Loss.

        Args:
            obs: [B, ...]
            actions: [B]
            old_log_probs: [B]
            returns: [B] (Target values for Value function)
            advantages: [B] (Normalized advantages)
            action_masks: [B, ActionSpace]
        """

        # 1. Forward Pass
        logits, values = self.model(obs)
        values = values.squeeze()

        if action_masks is not None:
            logits = logits.masked_fill(~action_masks, -float('inf'))

        log_probs_all = F.log_softmax(logits, dim=-1)
        probs_all = torch.exp(log_probs_all)

        # Gather log_probs for specific actions
        new_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. Policy Loss (Clipped Surrogate)
        ratio = torch.exp(new_log_probs - old_log_probs)
        # Safety clamp to prevent infinite ratios if old_log_prob is very small
        ratio = torch.clamp(ratio, 0.0, 100.0)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # 3. Value Loss
        value_loss = 0.5 * F.mse_loss(values, returns)

        # 4. Entropy Bonus
        # For masked actions, entropy should effectively be over valid actions only
        # H(pi) = - sum pi * log pi
        if action_masks is not None:
            # Masked entropy: sum(p * log(p)) only for valid actions.
            # p=0, log(p)=-inf -> p*log(p) = NaN. catch this.
            # safe_log_probs: replace -inf with 0.0 just for multiplication (result will be 0 anyway because prob is 0)
            safe_log_probs = torch.where(action_masks, log_probs_all, torch.zeros_like(log_probs_all))
            entropy = -torch.sum(probs_all * safe_log_probs, dim=-1).mean()
        else:
            entropy = -torch.sum(probs_all * log_probs_all, dim=-1).mean()

        total_loss = policy_loss + self.config.value_coef * value_loss - self.config.entropy_coef * entropy

        return {
            "loss": total_loss,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }

    @staticmethod
    def compute_gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        gamma: float = 0.99,
        lam: float = 0.95,
        next_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute GAE and Returns for a single trajectory.

        Args:
            rewards: [T]
            values: [T]
            dones: [T]
            next_value: Value of the next state after the last step

        Returns:
            advantages: [T]
            returns: [T]
        """
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0

        # Append next value for calculation
        values_next = torch.cat([values[1:], torch.tensor([next_value], device=values.device)])

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values_next[t] * (1 - dones[t]) - values[t]
            last_gae_lam = delta + gamma * lam * (1 - dones[t]) * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values
        return advantages, returns
