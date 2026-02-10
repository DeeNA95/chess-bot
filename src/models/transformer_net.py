import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class _ChessTransformerBase(nn.Module):
    '''
    Shared backbone for chess transformer variants.

    Architecture:
    1. CNN stem extracts local features → (B, embed_dim, 8, 8)
    2. Flatten to sequence of 64 square embeddings → (B, 64, embed_dim)
    3. Add 2D rank/file positional embeddings
    4. Transformer encoder for global attention
    5. Value head (scalar)

    Subclasses must implement a policy head.

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

        # 2D positional embeddings: separate rank and file
        self.rank_embeddings = nn.Parameter(
            torch.randn(1, 8, embed_dim) * 0.02
        )
        self.file_embeddings = nn.Parameter(
            torch.randn(1, 8, embed_dim) * 0.02
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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        '''Shared stem + transformer encoding. Returns (B, 64, embed_dim).'''
        features = self.stem(x)
        features = features.flatten(2).transpose(1, 2)

        pos = self.rank_embeddings[:, :, None, :] + self.file_embeddings[:, None, :, :]
        features = features + pos.reshape(1, 64, self.embed_dim)

        features = self.transformer(features)
        features = self.norm(features)
        return features

    def count_parameters(self) -> int:
        '''Returns total number of trainable parameters.'''
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ChessTransformerNet(_ChessTransformerBase):
    '''
    Hybrid CNN + Transformer with flat linear policy head.

    Policy: flatten all 64 square embeddings → linear → 4096 logits.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_head = nn.Linear(self.embed_dim * self.num_squares, 4096)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)

        policy_logits = self.policy_head(features.flatten(1))  # (B, 4096)

        value = self.value_head(features.mean(dim=1))  # (B, 1)

        return policy_logits, value


class ChessTransformerNetV2(_ChessTransformerBase):
    '''
    Hybrid CNN + Transformer with from-to attention policy head.

    Policy is computed as attention between from-squares and to-squares,
    giving more structured inductive bias for move prediction.
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.from_query = nn.Linear(self.embed_dim, self.embed_dim)
        self.to_key = nn.Linear(self.embed_dim, self.embed_dim)
        self.policy_scale = self.embed_dim ** -0.5
        self._init_weights()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)

        # Policy: from-to attention
        from_q = self.from_query(features)  # (B, 64, embed_dim)
        to_k = self.to_key(features)  # (B, 64, embed_dim)
        policy_logits = torch.bmm(from_q, to_k.transpose(1, 2)) * self.policy_scale
        policy_logits = policy_logits.flatten(1)  # (B, 4096)

        value = self.value_head(features.mean(dim=1))  # (B, 1)

        return policy_logits, value
