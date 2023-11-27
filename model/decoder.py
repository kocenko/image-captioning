from typing import Optional
from collections import Counter
import math

import torch
from torch import nn
import numpy as np

from model.transformer_sublayers import SelfAttention, CrossAttention, FeedForward


class DecoderInput(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        max_caption_length: int,
        embeddings: int,
        pos_n: int = 1000,
        learnable_pos: bool = True,
    ):
        super().__init__()
        assert embeddings % 2 == 0, f"Embeddings dimension should be divisible by 2 to perform fast positional encoding"

        self.learnable_pos = learnable_pos
        self.token_embedding = nn.Embedding(vocabulary_size, embeddings)

        if self.learnable_pos:
            self.positional_encoding = nn.Embedding(max_caption_length, embeddings)
            self.register_buffer("sequence_indices", torch.arange(max_caption_length), persistent=False)
        else:
            # Calculating positional encoding based on the "Attention is All You Need"
            # Based on: https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
            sequence_indices = torch.arange(max_caption_length).unsqueeze(1)
            divisor_term = torch.exp(torch.arange(0, embeddings, 2).float() * (-math.log(pos_n) / embeddings))
            positional_encoding = torch.zeros(max_caption_length, embeddings)
            positional_encoding[:, 0::2] = torch.sin(sequence_indices * divisor_term)
            positional_encoding[:, 1::2] = torch.cos(sequence_indices * divisor_term)
            positional_encoding = positional_encoding.unsqueeze(0)
            self.register_buffer("positional_encoding", positional_encoding, persistent=False)

    def forward(self, caption: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        token_embedding = self.token_embedding(caption)
        token_embedding = torch.masked_fill(token_embedding, padding_mask[:, :, None], 0)

        if self.learnable_pos:
            positional_embeddings = self.positional_encoding(
                self.sequence_indices[: token_embedding.shape[1]]
            ).unsqueeze(0)
            token_embedding = token_embedding + positional_embeddings
        else:
            token_embedding = token_embedding + self.positional_encoding[:, : token_embedding.shape[1], :]

        return token_embedding


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
        self.self_attention = SelfAttention(embeddings, heads_num, dropout_rate)
        self.cross_attention = CrossAttention(embeddings, cross_attention_key_dim, heads_num, dropout_rate)
        self.feed_forward = FeedForward(embeddings, dropout_rate)

        # noinspection PyTypeChecker
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_caption_length, max_caption_length)) == 0,
            persistent=False,
        )

    def forward(self, image, caption, key_padding_mask):
        caption = self.self_attention(
            caption, self.causal_mask[: caption.shape[1], : caption.shape[1]], key_padding_mask
        )
        caption = self.cross_attention(caption, image)
        caption = self.feed_forward(caption)
        return caption


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
