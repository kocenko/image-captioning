import math
from collections import Counter
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Implementation of Multi-Head Attention which uses batches for parallelization.
    """

    def __init__(self, input_shapes: Tuple[int, int, int], mask_out: bool = False, **kwargs):
        super().__init__()
        embeddings_number = kwargs["embeddings_number"]
        context_length = kwargs["context_length"]
        dropout_rate = kwargs["dropout_rate"]
        heads_number = kwargs["heads_number"]
        device = kwargs["device"]

        query_input_shape, key_input_shape, value_input_shape = input_shapes

        self.heads_number = heads_number
        self.mask_out = mask_out
        self.queries_weights = nn.Linear(query_input_shape, embeddings_number, device=device)
        self.keys_weights = nn.Linear(key_input_shape, embeddings_number, device=device)
        self.values_weights = nn.Linear(value_input_shape, embeddings_number, device=device)
        self.register_buffer(
            "masking_triangle",
            torch.tril(torch.ones(context_length, context_length, device=device)).view(
                1, 1, context_length, context_length
            ),
        )
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(embeddings_number, embeddings_number, device=device)
        self.output_dropout = nn.Dropout(dropout_rate)

        self.last_attention_scores: Optional[torch.Tensor] = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: Optional[torch.Tensor] = None):
        if value is None:
            value = key

        key_B, key_T, key_C = key.shape
        query_B, query_T, query_C = query.shape

        # Output shapes: [B, heads_num, T, heads_size]
        query_vector = self.queries_weights(query)
        key_vector = self.keys_weights(key)
        value_vector = self.values_weights(value)

        # Reshaping
        query_vector = query_vector.view(query_B, query_T, self.heads_number, -1).transpose(1, 2)
        if self.mask_out:
            np.save('../numpy_logs/torch_query.npy', query_vector.detach().cpu().numpy())
        key_vector = key_vector.view(key_B, key_T, self.heads_number, -1).transpose(1, 2)
        value_vector = value_vector.view(key_B, key_T, self.heads_number, -1).transpose(1, 2)

        # Affinities shape: [B, heads_num, T, T]
        affinities = query_vector @ key_vector.transpose(-2, -1)
        affinities /= math.sqrt(key_C)

        if self.mask_out:
            affinities = affinities.masked_fill(self.masking_triangle[:, :, key_T, :key_T] == 0, float("-inf"))
        affinities = F.softmax(affinities, dim=-1)
        if self.training:
            affinities = self.attention_dropout(affinities)
        self.last_attention_scores = affinities

        output = affinities @ value_vector  # [B, heads_num, T, head_size]
        output = output.transpose(1, 2).contiguous().view(query_B, query_T, query_C)  # [B, T, C]
        output = self.output_dropout(self.output_projection(output))

        return output


class TransformerBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        embeddings_number = kwargs["embeddings_number"]
        dropout_rate = kwargs["dropout_rate"]
        image_channels = kwargs["image_channels"]
        device = kwargs["device"]

        self_input_shapes = (embeddings_number, embeddings_number, embeddings_number)  # Q, K, V
        cross_input_shapes = (embeddings_number, image_channels, image_channels)  # Q, K, V

        # Self attention
        self.self_attention = MultiHeadAttention(self_input_shapes, mask_out=True, **kwargs)
        self.layer_normalization_1 = nn.LayerNorm(embeddings_number, device=device)

        # Cross attention
        self.cross_attention = MultiHeadAttention(cross_input_shapes, **kwargs)
        self.layer_normalization_2 = nn.LayerNorm(embeddings_number, device=device)

        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(embeddings_number, 4 * embeddings_number, device=device),
            nn.ReLU(),
            nn.Linear(4 * embeddings_number, embeddings_number, device=device),
            nn.Dropout(dropout_rate),
        )
        self.layer_normalization_3 = nn.LayerNorm(embeddings_number, device=device)
        self.last_attention_scores: Optional[torch.Tensor] = None

    def forward(self, image, caption):
        # Note: pre-norm formulation can be used
        sa = self.self_attention(caption, caption)
        np.save('../numpy_logs/torch_self_attention.npy', sa.detach().cpu().numpy())
        np.save('../numpy_logs/torch_attention_scores.npy', self.self_attention.last_attention_scores.detach().cpu().numpy())
        x = torch.add(caption, self.self_attention(caption, caption))
        x = self.layer_normalization_1(x)

        mock_image = torch.ones((32, 49, 576)).to(torch.float).to("cuda")
        mock_x = torch.ones((32, 20, 256)).to(torch.float).to("cuda")
        cross_attention = self.cross_attention(mock_x, mock_image)
        self.last_attention_scores = self.cross_attention.last_attention_scores
        np.save('../numpy_logs/torch_cross_attention.npy', cross_attention.detach().cpu().numpy())
        np.save('../numpy_logs/torch_cross_attention_scores.npy', self.cross_attention.last_attention_scores.detach().cpu().numpy())
        x = torch.add(x, cross_attention)
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
        # np.save('../numpy_logs/torch_flat_image.npy', image.detach().cpu().numpy())

        x = self.embedding(caption)
        # np.save('../numpy_logs/torch_embedding.npy', x.detach().cpu().numpy())


        for block in self.blocks:
            x = block(image, x)

        logits = self.output_layer(x)

        return logits
