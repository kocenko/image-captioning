from typing import Optional
from collections import Counter

import numpy as np
import torch
from torch import nn


class DecoderOutput(nn.Module):
    def __init__(
        self,
        embeddings: int,
        vocabulary_size: int,
        device: str,
        add_bias: bool = False,
        counter: Optional[Counter] = None,
        encode_map: Optional[dict] = None,
        banned_tokens: Optional[list[str]] = None,
    ):
        super().__init__()
        self.projection_to_vocabulary = nn.Linear(embeddings, vocabulary_size, device=device)

        bias = torch.zeros(vocabulary_size, device=device)
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
            bias = torch.tensor(bias, device=device)

        self.register_buffer("bias", bias)

    def forward(self, decoder_output: torch.Tensor) -> torch.Tensor:
        x = self.projection_to_vocabulary(decoder_output)
        x = x + self.bias
        x = x.transpose(-2, -1)  # Expected output shape: [B, vocabulary_size, T]
        return x
