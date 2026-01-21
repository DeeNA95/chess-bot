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

        # Stability: Compute log_probs in float32
        # FP16/BF16 softmax/log_softmax can be unstable with large logits
        log_probs_all = F.log_softmax(logits.to(torch.float32), dim=-1).to(logits.dtype)
        probs_all = torch.exp(log_probs_all)

        # Gather log_probs for specific actions
        new_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. Policy Loss (Clipped Surrogate)
        ratio = torch.exp(new_log_probs - old_log_probs)
        # Safety clamp to prevent infinite ratios if old_log_prob is very small
        ratio = torch.clamp(ratio, 0.0, 100.0)

        # Verify no NaNs in ratio
        if torch.isnan(ratio).any():
             # Fallback or zero out NaNs to prevent crash, though we should investigate why
             ratio = torch.nan_to_num(ratio, nan=1.0, posinf=100.0, neginf=0.0)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages

        policy_loss = -torch.min(surr1, surr2).mean()

        # 3. Value Loss
        value_loss = 0.5 * F.mse_loss(values, returns)

        if action_masks is not None:
            # Masked entropy: sum(p * log(p)) only for valid actions.
            # We use float32 versions for entropy calculation
            log_probs_f32 = log_probs_all.to(torch.float32)
            probs_f32 = probs_all.to(torch.float32)

            # Ensure valid log_probs for multiplication (mask out invalid ones)
            # For invalid actions, we want p * log(p) to be 0.
            # However, p will be 0 (exp(-inf)), log(p) will be -inf. 0*-inf = NaN.
            # We calculate the term, then sanitize it.

            # 1. Compute term p * log(p)
            entropy_terms = probs_f32 * log_probs_f32

            # 2. Mask out invalid actions (force them to 0.0)
            entropy_terms = torch.where(action_masks, entropy_terms, torch.zeros_like(entropy_terms))

            # 3. Sanitize any remaining NaNs (e.g. from 0*-inf inside valid masks if that happened, or just to be safe)
            # This handles the 0*-inf case even if it slipped through.
            entropy_terms = torch.nan_to_num(entropy_terms, nan=0.0)

            entropy = -torch.sum(entropy_terms, dim=-1).mean()
        else:
            entropy_terms = probs_all * log_probs_all
            entropy_terms = torch.nan_to_num(entropy_terms, nan=0.0)
            entropy = -torch.sum(entropy_terms, dim=-1).mean()

        if torch.isnan(entropy):
             entropy = torch.tensor(0.0, device=entropy.device)

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
