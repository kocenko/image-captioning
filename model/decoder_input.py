import math
import torch
from torch import nn


class DecoderInput(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        max_caption_length: int,
        embeddings: int,
        device: str,
        padding_idx: int = 0,
        pos_n: int = 1000
    ):
        super().__init__()
        assert embeddings % 2 == 0, f'Embeddings dimension should be divisible by 2 ot perform fast positional encoding'

        self.token_embedding = nn.Embedding(vocabulary_size, embeddings, padding_idx=padding_idx, device=device)

        # Calculating positional encoding based on the "Attention is All You Need"
        # Based on: https://medium.com/@hunter-j-phillips/positional-encoding-7a93db4109e6
        sequence_indices = torch.arange(max_caption_length).unsqueeze(1)
        divisor_term = torch.exp(torch.arange(0, embeddings, 2) * -(math.log(pos_n) / embeddings))
        positional_encoding = torch.zeros(max_caption_length, embeddings, device=device)
        positional_encoding[:, 0::2] = torch.sin(sequence_indices * divisor_term)
        positional_encoding[:, 1::2] = torch.cos(sequence_indices * divisor_term)
        self.register_buffer('positional_encoding', positional_encoding)

    def forward(self, caption: torch.Tensor) -> torch.Tensor:
        token_embedding = self.token_embedding(caption)
        return token_embedding + self.positional_encoding
