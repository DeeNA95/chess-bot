import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional, Tuple, List, Dict
from src.core.config import AppConfig

class GRPO:
    """
    Group Relative Policy Optimization (GRPO).

    Paper: DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models

    Key mechanisms:
    1. Sample a group of outputs {o_1, ... o_G} for the same input q.
    2. Compute rewards {r_1, ... r_G}.
    3. Compute advantages A_i = (r_i - mean(r)) / std(r).
    4. Optimization objective with KL penalty.
    """
    def __init__(self, config: AppConfig, model: nn.Module):
        self.config = config.grpo
        self.device = config.training.device or 'cpu'
        self.model = model
        self.ref_model = copy.deepcopy(model)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

    def update_ref_model(self):
        """Update reference model to current model weights."""
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()

    def compute_loss(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        rewards: torch.Tensor,
        action_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GRPO Loss.

        Args:
            obs: [B, C, H, W] observation tensor
            actions: [B] action indices taken
            old_log_probs: [B] log probability of actions under policy when sampled
            rewards: [B] rewards for each action (already computed group-wise or raw)
            action_masks: [B, ActionSpace] boolean mask

        Note: The batch B is expected to be Num_Questions * Group_Size.
              Grouping logic should happen before or within advantage computation.
              Here we assume 'rewards' are passed raw and we normalize them,
              OR 'rewards' are already advantages.

              For standard GRPO, we compute advantages within the group.
              So we assume the batch is ordered: [Q1_O1, Q1_O2, ..., Q2_O1, ...]
        """
        group_size = self.config.group_size
        assert obs.size(0) % group_size == 0, f"Batch size {obs.size(0)} not divisible by group size {group_size}"

        # 1. Compute Advantages (Group Relative)
        # Reshape rewards to [Num_Groups, Group_Size]
        num_groups = obs.size(0) // group_size

        # Ensure rewards are detached
        rewards = rewards.detach()

        rewards_grouped = rewards.view(num_groups, group_size)
        mean_r = rewards_grouped.mean(dim=1, keepdim=True)
        std_r = rewards_grouped.std(dim=1, keepdim=True) + 1e-8

        advantages = (rewards_grouped - mean_r) / std_r
        advantages = advantages.view(-1) # Flatten back to [B]

        # 2. Forward pass (Current Policy)
        # We need logits for the specific actions taken
        logits, _ = self.model(obs)
        if action_masks is not None:
            logits = logits.masked_fill(~action_masks, -float('inf'))

        log_probs_all = F.log_softmax(logits, dim=-1)

        # Gather log_probs for the specific actions taken
        # actions: [B], log_probs_all: [B, 4096]
        new_log_probs = log_probs_all.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 3. KL Divergence with Reference Model
        with torch.no_grad():
            ref_logits, _ = self.ref_model(obs)
            if action_masks is not None:
                ref_logits = ref_logits.masked_fill(~action_masks, -float('inf'))
            ref_log_probs_all = F.log_softmax(ref_logits, dim=-1)

        # KL(pi || pi_ref) = sum(pi * (log_pi - log_ref))
        probs_all = torch.exp(log_probs_all)
        kl_div_all = probs_all * (log_probs_all - ref_log_probs_all)

        if action_masks is not None:
            kl_div_all = kl_div_all.masked_fill(~action_masks, 0.0)

        kl_div = torch.sum(kl_div_all, dim=-1)

        # 4. Surrogate Objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * advantages

        # Loss = - (min(surr1, surr2) - beta * KL)
        # We minimize loss, so negate the objective
        pg_loss = -torch.min(surr1, surr2)
        kl_penalty = self.config.beta_kl * kl_div

        loss = (pg_loss + kl_penalty).mean()

        return {
            "loss": loss,
            "pg_loss": pg_loss.mean(),
            "kl_div": kl_div.mean(),
            "advantages": advantages.mean()
        }
