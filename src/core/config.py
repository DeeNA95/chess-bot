import yaml
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class MCTSSettings:
    num_simulations: int = 50
    c_puct: float = 1.5
    temperature: float = 1.0

@dataclass
class GRPOSettings:
    group_size: int = 16
    epsilon: float = 0.2
    beta_kl: float = 0.01
    entropy_coef: float = 0.01

@dataclass
class PPOSettings:
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    gae_lambda: float = 0.95
    gamma: float = 0.99
    ppo_epochs: int = 4

@dataclass
class TrainingSettings:
    total_games: int = 10000
    batch_size: int = 256
    games_per_update: int = 32
    lr: float = 0.0001
    checkpoint_dir: str = "/checkpoints"
    buffer_capacity: int = 100000
    num_parallel_games: int = 16
    device: Optional[str] = None

@dataclass
class AppConfig:
    algorithm: Literal["mcts", "grpo", "ppo"] = "mcts"
    training: TrainingSettings = field(default_factory=TrainingSettings)
    mcts: MCTSSettings = field(default_factory=MCTSSettings)
    grpo: GRPOSettings = field(default_factory=GRPOSettings)
    ppo: PPOSettings = field(default_factory=PPOSettings)

    @classmethod
    def load(cls, path: str = "config.yaml") -> "AppConfig":
        if not os.path.exists(path):
            print(f"Warning: Config file {path} not found. Using defaults.")
            return cls()

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        # Helper to recursively update dataclasses
        config = cls()

        if "algorithm" in data:
            config.algorithm = data["algorithm"]

        if "training" in data:
            _update_from_dict(config.training, data["training"])

        if "mcts" in data:
            _update_from_dict(config.mcts, data["mcts"])

        if "grpo" in data:
            _update_from_dict(config.grpo, data["grpo"])

        if "ppo" in data:
            _update_from_dict(config.ppo, data["ppo"])

        return config

def _update_from_dict(obj, data: dict):
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
