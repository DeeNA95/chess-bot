from abc import ABC, abstractmethod
import numpy as np
import torch
from typing import Dict, Any

class ChessAgent(ABC):
    """
    Abstract Base Class for all Chess Agents.
    Supports Model Agnostic design (PPO, DQN, AlphaZero).
    """

    @abstractmethod
    def predict(self, observation: Dict[str, Any], deterministic: bool = False) -> int:
        """
        Given an observation (tensor + mask), return the action index (0-4095).
        """
        pass

    @abstractmethod
    def train_step(self, batch: Any) -> Dict[str, float]:
        """
        Perform a single training step on a batch of data.
        Returns metrics (loss, etc).
        """
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def load(self, path: str):
        pass
