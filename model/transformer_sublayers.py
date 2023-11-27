from typing import Optional

import torch
import torch.nn as nn

from model.multihead_attention import MultiHeadAttention


class LocalitySelfAttention(nn.Module):
    def __init__(self, embeddings: int, heads_num: int, dropout_rate: float):
        super().__init__()
        self.mha = MultiHeadAttention(
            input_shapes=(embeddings, embeddings, embeddings),
            embeddings_number=embeddings,
            heads_number=heads_num,
            dropout_rate=dropout_rate,
            trainable_scale=True,
        )
        self.layer_norm_before = nn.LayerNorm(embeddings)
        self.layer_norm_after = nn.LayerNorm(embeddings)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention = self.mha(x, x, x, self.layer_norm_before(attention_mask))
        residual = x + attention
        normalized = self.layer_norm_after(residual)
        return normalized


class SelfAttention(nn.Module):
    def __init__(self, embeddings: int, heads_num: int, dropout_rate: float):
        super().__init__()
        self.mha = MultiHeadAttention(
            input_shapes=(embeddings, embeddings, embeddings),
            embeddings_number=embeddings,
            heads_number=heads_num,
            dropout_rate=dropout_rate,
        )
        self.layer_norm = nn.LayerNorm(embeddings)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention = self.mha(x, x, x, attention_mask, key_padding_mask)
        residual = x + attention
        normalized = self.layer_norm(residual)
        return normalized


class CrossAttention(nn.Module):
    def __init__(self, embeddings: int, key_dim: int, heads_num: int, dropout_rate: float):
        super().__init__()
        self.mha = MultiHeadAttention(
            input_shapes=(embeddings, key_dim, key_dim),
            embeddings_number=embeddings,
            heads_number=heads_num,
            dropout_rate=dropout_rate,
        )
        self.layer_norm = nn.LayerNorm(embeddings)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attention = self.mha(x, y, y, attention_mask, key_padding_mask)
        residual = x + attention
        normalized = self.layer_norm(residual)
        return normalized


class FeedForward(nn.Module):
    def __init__(self, embeddings: int, hidden_scale: int, dropout_rate: float):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(embeddings, hidden_scale * embeddings),
            nn.ReLU(),
            nn.Linear(hidden_scale * embeddings, embeddings),
            nn.Dropout(dropout_rate),
        )
        self.layer_norm = nn.LayerNorm(embeddings)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fed = self.ff(x)
        residual = x + fed
        normalized = self.layer_norm(residual)
        return normalized
