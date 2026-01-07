import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, buffer_size, obs_shape, action_mask_shape, device="cpu"):
        self.obs = torch.zeros((buffer_size, *obs_shape), dtype=torch.float32, device=device)
        self.actions = torch.zeros(buffer_size, dtype=torch.long, device=device)
        self.logprobs = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.values = torch.zeros(buffer_size, dtype=torch.float32, device=device)
        self.masks = torch.zeros((buffer_size, *action_mask_shape), dtype=torch.bool, device=device)

        self.pos = 0
        self.buffer_size = buffer_size
        self.device = device

    def add(self, obs, action, logprob, reward, done, value, mask):
        self.obs[self.pos] = obs
        self.actions[self.pos] = action
        self.logprobs[self.pos] = logprob
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
        self.values[self.pos] = value
        self.masks[self.pos] = mask

        self.pos += 1

    def full(self):
        return self.pos >= self.buffer_size

    def reset(self):
        self.pos = 0

    def get(self):
        # Returns all data
        # Note: We might want to handle incomplete buffers or check if full?
        # For now assume we always fill.
        return (
            self.obs,
            self.actions,
            self.logprobs,
            self.rewards,
            self.dones,
            self.values,
            self.masks
        )
