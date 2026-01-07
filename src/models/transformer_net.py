import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChessTransformerNet(nn.Module):
    '''
    Hybrid CNN + Transformer network for chess.

    Architecture:
    1. CNN stem extracts local features → (B, embed_dim, 8, 8)
    2. Flatten to sequence of 64 square embeddings → (B, 64, embed_dim)
    3. Add learnable positional embeddings
    4. Transformer encoder for global attention
    5. Dual heads: Policy (4096 logits) + Value (scalar)

    Input: (B, num_planes, 8, 8) - canonical board representation
    Output: (policy_logits, value)
    '''

    def __init__(
        self,
        num_input_planes: int = 20,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        action_space: int = 4096
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_squares = 64

        # CNN stem: extract local features from board planes
        self.stem = nn.Sequential(
            nn.Conv2d(num_input_planes, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.GELU()
        )

        # Learnable positional embeddings for each of 64 squares
        self.positional_embeddings = nn.Parameter(
            torch.randn(1, self.num_squares, embed_dim) * 0.02
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False
        )

        # Layer norm before heads
        self.norm = nn.LayerNorm(embed_dim)

        # Policy head: from-to attention mechanism
        self.policy_from_proj = nn.Linear(embed_dim, embed_dim // 2)
        self.policy_to_proj = nn.Linear(embed_dim, embed_dim // 2)
        self.policy_head = nn.Linear(embed_dim * self.num_squares, action_space)

        # Value head: pooled representation → scalar
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Value in [-1, 1]
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        '''
        Forward pass.

        Args:
            x: (B, num_planes, 8, 8) board tensor

        Returns:
            policy_logits: (B, 4096) raw logits for from-to moves
            value: (B, 1) estimated value of position
        '''
        batch_size = x.shape[0]

        # CNN stem: (B, num_planes, 8, 8) → (B, embed_dim, 8, 8)
        features = self.stem(x)

        # Reshape to sequence: (B, embed_dim, 8, 8) → (B, 64, embed_dim)
        features = features.flatten(2).transpose(1, 2)

        # Add positional embeddings
        features = features + self.positional_embeddings

        # Transformer: (B, 64, embed_dim) → (B, 64, embed_dim)
        features = self.transformer(features)
        features = self.norm(features)

        # Policy: flatten all squares and project to action space
        policy_features = features.flatten(1)  # (B, 64 * embed_dim)
        policy_logits = self.policy_head(policy_features)  # (B, 4096)

        # Value: mean pool across squares
        value_features = features.mean(dim=1)  # (B, embed_dim)
        value = self.value_head(value_features)  # (B, 1)

        return policy_logits, value

    def count_parameters(self) -> int:
        '''Returns total number of trainable parameters.'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ChessTransformerNetV2(ChessTransformerNet):
    '''
    Enhanced version with from-to attention for policy head.

    Policy is computed as attention between from-squares and to-squares,
    giving more structured inductive bias for move prediction.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Override policy head with attention-based mechanism
        embed_dim = self.embed_dim
        self.from_query = nn.Linear(embed_dim, embed_dim)
        self.to_key = nn.Linear(embed_dim, embed_dim)
        self.policy_scale = embed_dim ** -0.5

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]

        # CNN stem + transformer (same as parent)
        features = self.stem(x)
        features = features.flatten(2).transpose(1, 2)
        features = features + self.positional_embeddings
        features = self.transformer(features)
        features = self.norm(features)

        # Policy: from-to attention
        # Each square can be a "from" square (query) or "to" square (key)
        from_q = self.from_query(features)  # (B, 64, embed_dim)
        to_k = self.to_key(features)  # (B, 64, embed_dim)

        # Compute attention scores: (B, 64, 64) → flatten to (B, 4096)
        policy_logits = torch.bmm(from_q, to_k.transpose(1, 2)) * self.policy_scale
        policy_logits = policy_logits.flatten(1)  # (B, 4096)

        # Value: mean pool
        value_features = features.mean(dim=1)
        value = self.value_head(value_features)

        return policy_logits, value
