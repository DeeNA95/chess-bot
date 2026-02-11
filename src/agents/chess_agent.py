import torch
import torch.amp
from src.models.transformer_net import ChessTransformerNetV2
from src.core.config import ModelSettings
from typing import Dict, Any, Optional
from ..utils import get_device

class ChessAgent:
    def __init__(self, device="cpu", lr=5e-5, weight_decay=1e-3, model_config: Optional[ModelSettings] = None):
        self.device = get_device()
        mc = model_config or ModelSettings()
        self.model = ChessTransformerNetV2(
            num_input_planes=116,
            embed_dim=mc.embed_dim,
            num_layers=mc.num_layers,
            num_heads=mc.num_heads,
            mlp_ratio=mc.mlp_ratio,
            dropout=mc.dropout,
        ).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.scaler = torch.amp.GradScaler('cuda') if device == 'cuda' else None

        # LR Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10000, T_mult=2
        )

    def predict(self, observation: Dict[str, Any], deterministic: bool = False) -> int:
        """Simple model inference without MCTS."""

        obs = observation['observation']

        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)

        obs = obs.unsqueeze(0).to(self.device)
        mask = observation['action_mask'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                logits, _ = self.model(obs)
                logits[~mask] = -float('inf')
                probs = torch.softmax(logits, dim=-1)
                if deterministic:
                    return int(torch.argmax(probs).item())
                return int(torch.distributions.Categorical(probs).sample().item())

    def train_step(self, batch: Any) -> Dict[str, float]:
        """Legacy PPO train_step - not used in AlphaZero loop."""
        return {}

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
