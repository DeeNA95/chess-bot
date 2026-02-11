import yaml
import os
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class ModelSettings:
    embed_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.1

@dataclass
class MCTSSettings:
    num_simulations: int = 50
    c_puct: float = 1.5
    temperature: float = 1.0
    dirichlet_alpha: float = 0.03
    dirichlet_epsilon: float = 0.0
    reuse_tree: bool = True
    leaves_per_sim: int = 8
    max_nodes_per_tree: int = 100000

@dataclass
class GRPOSettings:
    group_size: int = 16
    epsilon: float = 0.2
    beta_kl: float = 0.01
    entropy_coef: float = 0.01

@dataclass
class RewardSettings:
    stockfish_path: str = "/opt/homebrew/bin/stockfish"
    stockfish_weight: float = 0.8
    material_weight: float = 0.1
    outcome_weight: float = 0.1
    stockfish_depth: int = 8
    stockfish_hash: int = 64
    num_workers: int = 4

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
class SelfPlaySettings:
    num_workers: int = 0
    games_per_worker: int = 4
    max_moves: int = 100
    flush_every_moves: int = 10
    sync_weights_every: int = 1

@dataclass
class AppConfig:
    algorithm: Literal["mcts", "grpo", "ppo", "ppo_mcts", "grpo_mcts"] = "mcts"
    model: ModelSettings = field(default_factory=ModelSettings)
    training: TrainingSettings = field(default_factory=TrainingSettings)
    mcts: MCTSSettings = field(default_factory=MCTSSettings)
    grpo: GRPOSettings = field(default_factory=GRPOSettings)
    ppo: PPOSettings = field(default_factory=PPOSettings)
    rewards: RewardSettings = field(default_factory=RewardSettings)
    self_play: SelfPlaySettings = field(default_factory=SelfPlaySettings)

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

        if "model" in data:
            _update_from_dict(config.model, data["model"])

        if "training" in data:
            _update_from_dict(config.training, data["training"])

        if "mcts" in data:
            _update_from_dict(config.mcts, data["mcts"])

        if "grpo" in data:
            _update_from_dict(config.grpo, data["grpo"])

        if "ppo" in data:
            _update_from_dict(config.ppo, data["ppo"])

        if "rewards" in data:
            _update_from_dict(config.rewards, data["rewards"])
        if "self_play" in data:
            _update_from_dict(config.self_play, data["self_play"])

        return config

def _update_from_dict(obj, data: dict):
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, value)
