import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from src.agents.base_agent import ChessAgent
from src.models.transformer_net import ChessTransformerNet
from typing import Dict, Any

class PPOAgent(ChessAgent):
    def __init__(self, device="cpu", lr=5e-5):
        self.device = device
        # Use new Transformer Net with 116 planes
        self.model = ChessTransformerNet(num_input_planes=116).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Mixed Precision Scaler
        # Use torch.amp.GradScaler for compatibility
        self.scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10000, T_mult=2
        )

        self.gamma = 0.99
        self.eps_clip = 0.2
        self.K_epochs = 4

    def predict(self, observation: Dict[str, Any], deterministic: bool = False) -> int:
        obs = observation['observation']
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        obs = obs.unsqueeze(0).to(self.device)

        mask = observation['action_mask']
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.bool)
        mask = mask.unsqueeze(0).to(self.device)

        with torch.no_grad():
            # Autocast for inference (optional but good for consistency)
            # Check for bf16 support dynamically
            dtype = torch.float16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16

            # Use torch.amp.autocast
            with torch.amp.autocast('cuda', enabled=(self.device=='cuda'), dtype=dtype):
                logits, _ = self.model(obs)
                logits[~mask] = -float('inf')
                probs = F.softmax(logits, dim=-1)

            if deterministic:
                action = int(torch.argmax(probs).item())
            else:
                dist = torch.distributions.Categorical(probs)
                action = int(dist.sample().item())

        return action

    def get_action_and_value(self, obs, mask, action=None, deterministic=False):
        # Batch version for training

        # Determine dtype for autocast
        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

        with torch.amp.autocast('cuda', enabled=(self.device=='cuda'), dtype=dtype):
            logits, value = self.model(obs)
            logits[~mask] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            if action is None:
                if deterministic:
                    action = torch.argmax(probs, dim=-1)
                else:
                    action = dist.sample()

            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

        return action, log_prob, entropy, value

    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Perform PPO update using the collected rollout buffer.
        """
        buffer = batch
        batch_size = 128
        obs, actions, old_logprobs, rewards, dones, values, masks = buffer.get()

        # Calculate Advantages (GAE)
        advantages = torch.zeros_like(rewards).to(self.device)
        last_gae_lam = 0
        gamma = self.gamma
        lam = 0.95

        num_steps = len(rewards)

        # GAE moves backwards
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_non_terminal = 1.0 - dones[t]
                next_values = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_values = values[t+1]

            delta = rewards[t] + gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = {"loss": 0.0, "actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0}

        # Determine dtype for autocast
        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

        for _ in range(self.K_epochs):
            indices = torch.randperm(num_steps).to(self.device)

            for start in range(0, num_steps, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]

                b_obs = obs[batch_idx]
                b_actions = actions[batch_idx]
                b_old_logprobs = old_logprobs[batch_idx]
                b_advantages = advantages[batch_idx]
                b_returns = returns[batch_idx]
                b_masks = masks[batch_idx]

                # Mixed Precision Training
                with torch.amp.autocast('cuda', enabled=(self.device=='cuda'), dtype=dtype):
                    logits, state_values = self.model(b_obs)
                    state_values = state_values.squeeze()

                    logits[~b_masks] = -float('inf')
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)

                    new_logprobs = dist.log_prob(b_actions)
                    dist_entropy = dist.entropy().mean()

                    ratio = torch.exp(new_logprobs - b_old_logprobs)

                    surr1 = ratio * b_advantages
                    surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * b_advantages

                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(state_values, b_returns)

                    loss = actor_loss + 0.5 * critic_loss - 0.02 * dist_entropy

                self.optimizer.zero_grad()

                if self.scaler:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

                # Step Scheduler
                self.scheduler.step()

                metrics["loss"] = loss.item()
                metrics["actor_loss"] = actor_loss.item()
                metrics["critic_loss"] = critic_loss.item()
                metrics["entropy"] = dist_entropy.item()

        return metrics

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
