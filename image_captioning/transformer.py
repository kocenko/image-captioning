import math
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class SingleHeadAttention(nn.Module):
    def __init__(
        self,
        input_shapes: tuple[int, int, int],
        embeddings_number: int,
        dropout_rate: float,
        device: str,
        **kwargs
    ) -> None:
        super().__init__()

        self.device = device
        self.key_projection_dim = embeddings_number
        self.attention_score: Optional[torch.Tensor] = None

        # Key and query projections should have the same dimensions. Implied by embeddings_number.
        self.query_projection = nn.Linear(input_shapes[0], embeddings_number, bias=False, device=device)
        self.key_projection = nn.Linear(input_shapes[1], embeddings_number, bias=False, device=device)
        self.value_projection = nn.Linear(input_shapes[2], embeddings_number, bias=False, device=device)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: bool
    ):
        query = self.query_projection(query)  # [B, T_q, key_dim]
        key = self.key_projection(key)        # [B, T_k, key_dim]
        value = self.value_projection(value)  # [B, T_v, value_dim]

        affinity = query @ key.transpose(-2, -1)  # [B, T_q, T_k]
        affinity = affinity / math.sqrt(self.key_projection_dim)
        if mask:
            causal_mask = torch.tril(torch.ones(query.shape[1], key.shape[1], device=self.device)).unsqueeze(0)  # [1, T_q, T_k]
            affinity += (causal_mask == 0) * -1e9  # For numerical stability

        self.attention_score = self.softmax(affinity)
        self.attention_score = self.dropout(self.attention_score)

        attention = self.attention_score @ value  # [B, T_q, value_dim]
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_shapes: tuple[int, int, int],
        **kwargs
    ) -> None:
        super().__init__()

        heads_number = kwargs["heads_number"]
        embeddings_number = kwargs["embeddings_number"]
        dropout_rate = kwargs["dropout_rate"]
        device = kwargs["device"]

        self.heads = nn.ModuleList([SingleHeadAttention(input_shapes, **kwargs) for _ in range(heads_number)])
        self.projection = nn.Linear(heads_number * embeddings_number, embeddings_number, device=device)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: bool
    ) -> torch.Tensor:
        
        heads_outputs = []
        for head in self.heads:
            heads_outputs.append(head(query, key, value, mask).unsqueeze(1))

        x = torch.cat(heads_outputs, dim=-1)
        x = self.projection(x)
        x = x.squeeze(dim=1)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        embeddings = kwargs["embeddings_number"]
        channels = kwargs["image_channels"]
        dropout_rate = kwargs["dropout_rate"]
        device = kwargs["device"]

        # Self attention
        self.self_attention = MultiHeadAttention(input_shapes=(embeddings, embeddings, embeddings), **kwargs)
        self.layer_normalization_1 = nn.LayerNorm(embeddings, device=device)

        # Cross attention
        self.cross_attention = MultiHeadAttention(input_shapes=(embeddings, channels, channels), **kwargs)
        self.layer_normalization_2 = nn.LayerNorm(embeddings, device=device)

        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(embeddings, 2 * embeddings, device=device),
            nn.ReLU(),
            nn.Linear(2 * embeddings, embeddings, device=device),
            nn.Dropout(dropout_rate),
        )
        self.layer_normalization_3 = nn.LayerNorm(embeddings, device=device)

    def forward(self, image, caption):
        # Note: pre-norm formulation can be used
        x = torch.add(caption, self.self_attention(caption, caption, caption, True))
        x = self.layer_normalization_1(x)

        x = torch.add(x, self.cross_attention(x, image, image, False))
        x = self.layer_normalization_2(x)

        x = x + self.feed_forward(x)
        x = self.layer_normalization_3(x)

        return x


class TokenEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        embeddings_number = kwargs["embeddings_number"]
        vocabulary_size = kwargs["vocabulary_size"]
        context_length = kwargs["context_length"]
        self.device = kwargs["device"]

        self.token_embedding_table = nn.Embedding(vocabulary_size, embeddings_number, padding_idx=0, device=self.device)
        self.positional_embedding = nn.Embedding(context_length, embeddings_number, device=self.device)

    @staticmethod
    def __positional_embedding(batch_size: int, embedding_size: int) -> torch.Tensor:
        # Fixed positional embedding
        raise NotImplementedError

    def forward(self, sequence):
        _, sequence_size = sequence.shape
        token_embedding = self.token_embedding_table(sequence)
        positional_embedding = self.positional_embedding(torch.arange(sequence_size, device=self.device).unsqueeze(0))
        sequence = torch.add(token_embedding, positional_embedding)
        return sequence


class DecoderOutputLayer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.counter: Counter = kwargs["word_count"]
        self.encode_map: dict = kwargs["encode_map"]
        self.banned_tokens: list[int] = [self.encode_map[token] for token in kwargs["banned_tokens"]]
        self.device: str = kwargs["device"]

        embeddings_number = kwargs["embeddings_number"]
        vocabulary_size = kwargs["vocabulary_size"]
        self.linear = nn.Linear(embeddings_number, vocabulary_size, device=self.device)

        counts_list = np.zeros(shape=(vocabulary_size,))
        token_indexes = np.array([self.encode_map[key] for key in self.counter.keys()])
        counts_list[token_indexes] = list(self.counter.values())
        counts_list[self.banned_tokens] = 0

        # Creating bias based on the tokens distribution
        all_occurrences = counts_list.sum()
        scaled_counts = counts_list / all_occurrences  # p
        scaled_counts[counts_list == 0] = 1
        log_p = np.log(scaled_counts)

        self.bias = log_p
        self.bias[counts_list == 0] = -1e9
        self.bias = torch.tensor(self.bias, device=self.device)

    def forward(self, x):
        x = self.linear(x)
        return x + self.bias


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.flattener = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        # Expected input (B, C, H, W)
        x = self.flattener(x)
        x = x.transpose(-2, -1)
        return x


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.blocks_number = kwargs["blocks_number"]
        self.device = kwargs["device"]

        # Embeddings (with positional)
        self.image_flattener = EncoderBlock()
        self.embedding = TokenEmbedding(**kwargs)
        self.blocks = nn.ModuleList([TransformerBlock(**kwargs) for _ in range(self.blocks_number)])
        self.output_layer = DecoderOutputLayer(**kwargs)

    def forward(self, image, caption):
        image = self.image_flattener(image)

        x = self.embedding(caption)

        for block in self.blocks:
            x = block(image, x)

        logits = self.output_layer(x)

        return logits
