from typing import Optional
import torch
from torch import nn
import math


class MultiHeadAttention(nn.Module):
    """Multi-head Attention Layer

    This class implements multi-head attention mechanism.
    It allows the model to attend to different parts of the sequence independently and in parallel.
    Relative to the original implementation, it is extended by a learnable parameter tau.
    Parameter tau helps to sharpen the attention scores.

    Args:
        input_shapes (tuple): tuple of three values, each representing input dimensions of query, key and value
        embeddings_number (int): embeddings dimension
        heads_number (int): number of heads to parallelize attention
        trainable_scale (bool): whether to make tau trainable

    Attributes:
        num_heads (int): number of heads to parallelize attention
        key_dim (int): dimensions of key (and query) projections
        tau (int): parameter used for attention score scaling
        query_projection (nn.Linear): dense layer used for query projection
        key_projection (nn.Linear): dense layer used for key projection
        value_projection (nn.Linear): dense layer used for value projection
        softmax (nn.Softmax): softmax layer to apply on attention score
        output_projection (nn.Linear): dense layer to merge scores between attention heads
        attention_weights (torch.Tensor): attention weights between query and key projected vectors

    Methods:
        forward: To compute attention.

    Raises:
        AssertionError: If number of embeddings is not divisible by number of heads

    References:
        - "Attention is All You Need" (Vaswani et al., 2017) (https://arxiv.org/abs/1706.03762)
        - "Vision Transformer for Small-Size Datasets" (S. Lee, S. Lee and B. C. Song, 2022)
          (https://ieeexplore.ieee.org/document/9957006)
    """

    def __init__(
        self,
        input_shapes: tuple[int, int, int],
        embeddings_number: int,
        heads_number: int,
        dropout_rate: float,
        trainable_scale: bool = False,
    ) -> None:
        super().__init__()

        assert (
            embeddings_number % heads_number == 0
        ), f"Embeddings number: {embeddings_number} is not divisible by the number of heads: {heads_number}"

        self.num_heads = heads_number
        self.key_dim = embeddings_number // heads_number  # AKA head_dim

        tau = math.sqrt(self.key_dim)
        if trainable_scale:
            self.tau = nn.Parameter(torch.tensor(tau))
        else:
            self.tau = tau

        self.query_projection = nn.Linear(input_shapes[0], embeddings_number)
        self.key_projection = nn.Linear(input_shapes[1], embeddings_number)
        self.value_projection = nn.Linear(input_shapes[2], embeddings_number)
        self.output_projection = nn.Linear(embeddings_number, embeddings_number)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.attention_weights = None

    @staticmethod
    def _build_mask(
        attention_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        batches: int,
        heads: int,
        query_dim: int,
    ) -> Optional[torch.Tensor]:
        if attention_mask is None and padding_mask is None:
            return None
        if attention_mask is not None and padding_mask is None:
            output_mask = attention_mask.unsqueeze(0).unsqueeze(0)
            output_mask.expand(batches, heads, attention_mask.shape[0], attention_mask.shape[1])
            return output_mask
        if attention_mask is None and padding_mask is not None:
            output_mask = padding_mask.unsqueeze(1).unsqueeze(1)
            output_mask.expand(padding_mask.shape[0], heads, query_dim, padding_mask.shape[1])
            return output_mask
        if attention_mask is not None and padding_mask is not None:
            output_mask = attention_mask[:, None, :] | padding_mask[None, None, :]
            output_mask = output_mask.expand(heads, query_dim, batches, attention_mask.shape[1])
            output_mask = output_mask.permute(2, 0, 1, 3)
            return output_mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert (
            key.shape[1] == value.shape[1]
        ), f"Sequence sizes of key and value do not match - key: {key.shape[1]}, value: {value.shape[1]}."

        # Remembering input dimensions
        B, T_q, C_q = query.shape
        B, T_k, C_k = key.shape

        # Projecting inputs
        query = self.query_projection(query)  # [B, T_q, key_dim * num_heads]
        key = self.key_projection(key)  # [B, T_k, key_dim * num_heads]
        value = self.value_projection(value)  # [B, T_v, val_dim * num_heads]

        # Splitting heads
        query = query.reshape(B, T_q, self.num_heads, -1).permute(0, 2, 1, 3) / self.tau  # [B, num_heads, T_q, key_dim]
        key = key.reshape(B, T_k, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, num_heads, T_k, key_dim]
        value = value.reshape(B, T_k, self.num_heads, -1).permute(0, 2, 1, 3)  # [B, num_heads, T_v, val_dim]

        # Dot-product between the query and the key
        affinity = query @ key.transpose(-2, -1)  # [B, num_heads, T_q, T_k]

        # Masking sequence items that should not be attended to or are padding
        mask = self._build_mask(attention_mask, key_padding_mask, B, self.num_heads, T_q)
        if mask is not None:
            affinity += mask * -1e9

        affinity = self.softmax(affinity)
        self.attention_weights = affinity

        affinity = self.dropout(affinity)

        # Output score
        attention = affinity @ value  # [B, num_heads, T_v, val_dim]
        attention = attention.permute(0, 2, 1, 3).contiguous()
        attention = torch.flatten(attention, start_dim=2)

        # Output projection
        attention = self.output_projection(attention)

        return attention
