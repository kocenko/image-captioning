from typing import Optional
from collections import Counter
import math

import torch
from torch import nn
import numpy as np

from model.multihead_attention import MultiHeadAttention


class DecoderInput(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        max_caption_length: int,
        embeddings: int,
        padding_idx: int = 0,
        pos_n: int = 1000,
    ):
        super().__init__()
        assert embeddings % 2 == 0, f"Embeddings dimension should be divisible by 2 to perform fast positional encoding"

        self.padding_index = padding_idx
        self.token_embedding = nn.Embedding(vocabulary_size, embeddings)

        # Calculating positional encoding based on the "Attention is All You Need"
        # Based on: https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
        sequence_indices = torch.arange(max_caption_length).unsqueeze(1)
        divisor_term = torch.exp(torch.arange(0, embeddings, 2).float() * (-math.log(pos_n) / embeddings))
        positional_encoding = torch.zeros(max_caption_length, embeddings)
        positional_encoding[:, 0::2] = torch.sin(sequence_indices * divisor_term)
        positional_encoding[:, 1::2] = torch.cos(sequence_indices * divisor_term)
        positional_encoding = positional_encoding.unsqueeze(0)

        self.register_buffer("positional_encoding", positional_encoding, persistent=False)

    def forward(self, caption: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        padding_mask = torch.tensor(caption == self.padding_index)
        token_embedding = self.token_embedding(caption)
        token_embedding = torch.masked_fill(token_embedding, padding_mask[:, :, None], 0)
        token_embedding = token_embedding + self.positional_encoding[:, : token_embedding.shape[1], :]
        return token_embedding, padding_mask


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embeddings: int,
        cross_attention_key_dim: int,
        dropout_rate: float,
        heads_num: int,
        max_caption_length: int,
    ):
        super().__init__()
        self.self_attention_pre_normalization = nn.LayerNorm(embeddings)
        self.self_attention = MultiHeadAttention(
            input_shapes=(embeddings, embeddings, embeddings),
            embeddings_number=embeddings,
            heads_number=heads_num,
            dropout_rate=dropout_rate,
        )
        self.self_attention_post_normalization = nn.LayerNorm(embeddings)

        self.cross_attention_pre_normalization = nn.LayerNorm(cross_attention_key_dim)
        self.cross_attention = MultiHeadAttention(
            input_shapes=(embeddings, cross_attention_key_dim, cross_attention_key_dim),
            embeddings_number=embeddings,
            heads_number=heads_num,
            dropout_rate=dropout_rate,
        )
        self.cross_attention_post_normalization = nn.LayerNorm(embeddings)

        self.feed_forward = nn.Sequential(
            nn.Linear(embeddings, 2 * embeddings),
            nn.GELU(),
            nn.Linear(2 * embeddings, embeddings),
            nn.Dropout(dropout_rate),
        )

        # noinspection PyTypeChecker
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_caption_length, max_caption_length)) == 0,
            persistent=False,
        )

    def forward(self, image, caption, key_padding_mask):
        # caption_norm = self.self_attention_pre_normalization(caption)
        sa = self.self_attention(caption, caption, caption, self.causal_mask, key_padding_mask)
        x = caption + sa
        x_post_norm = self.self_attention_post_normalization(x)

        # image_norm = self.cross_attention_pre_normalization(image)
        cr = self.cross_attention(x_post_norm, image, image)
        x = x + cr
        x_post_norm = self.cross_attention_post_normalization(x)

        ff = self.feed_forward(x_post_norm)
        x = x + ff

        return x


class DecoderOutput(nn.Module):
    def __init__(
        self,
        embeddings: int,
        vocabulary_size: int,
        add_bias: bool = False,
        counter: Optional[Counter] = None,
        encode_map: Optional[dict] = None,
        banned_tokens: Optional[list[str]] = None,
    ):
        super().__init__()
        self.projection_to_vocabulary = nn.Linear(embeddings, vocabulary_size)

        bias = torch.zeros(vocabulary_size)
        if add_bias:
            assert not any(
                [counter is None, encode_map is None, banned_tokens is None]
            ), "Positional parameters should be specified if you want to add bias"

            banned_indices = [encode_map[token] for token in banned_tokens]
            token_indices = [encode_map[key] for key in counter.keys()]
            counts_list = np.zeros(vocabulary_size)
            counts_list[token_indices] = [*counter.values()]
            counts_list[banned_indices] = 0

            cumulative_frequency = counts_list.sum()
            scaled_counts_list = counts_list / cumulative_frequency
            scaled_counts_list[counts_list == 0] = 1  # To ensure that log of it is equal to 0
            log_counts = np.log(scaled_counts_list)

            bias = log_counts
            bias[counts_list == 0] = -1e9  # Masking banned or non-appearing tokens
            bias = torch.tensor(bias)

        self.register_buffer("bias", bias, persistent=False)

    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        x = self.projection_to_vocabulary(decoder_output)
        x = x + self.bias
        x = x.transpose(-2, -1)  # Expected output shape: [B, vocabulary_size, T]
        return x
