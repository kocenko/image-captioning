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

    def __init__(
        self,
        heads_number: int,
        embeddings_number: int,
        inputs_channels: tuple[int, int, int],
        dropout_rate: int,
        device: str,
        **kwargs,
    ):
        super().__init__()
        self.embeddings_number = embeddings_number
        self.heads_number = heads_number
        self.queries_weights = nn.Linear(inputs_channels[0], heads_number * embeddings_number, device=device)
        self.keys_weights = nn.Linear(inputs_channels[1], heads_number * embeddings_number, device=device)
        self.values_weights = nn.Linear(inputs_channels[2], heads_number * embeddings_number, device=device)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.output_projection = nn.Linear(heads_number * embeddings_number, embeddings_number, device=device)
        self.output_dropout = nn.Dropout(dropout_rate)
        self.last_attention_scores: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        if value is None:
            value = key

        B, query_T, query_C = query.shape
        key_B, key_T, key_C = key.shape

        # Output shapes: [B, T_x, heads_num, heads_size]
        query_vector = self.queries_weights(query).view(B, query_T, self.heads_number, -1) / math.sqrt(self.embeddings_number)
        key_vector = self.keys_weights(key).view(key_B, key_T, self.heads_number, -1)
        value_vector = self.values_weights(value).view(key_B, key_T, self.heads_number, -1)

        # if attention_mask is None:
        #     np.save("../numpy_logs/torch_query.npy", query_vector.detach().cpu().numpy())
        #     np.save("../numpy_logs/torch_key.npy", key_vector.detach().cpu().numpy())
        #     np.save("../numpy_logs/torch_value.npy", value_vector.detach().cpu().numpy())

        # Affinities shape: [B, heads_num, T_query, T_key]
        affinities = query_vector.transpose(1, 2) @ key_vector.permute(0, 2, 3, 1)

        if attention_mask is not None:
            affinities = affinities.masked_fill(attention_mask[:, :, :key_T, :key_T] == 0, float("-inf"))

        affinities = F.softmax(affinities, dim=-1)
        self.last_attention_scores = affinities
        # if attention_mask is None:
        #     np.save("../numpy_logs/torch_attention_scores.npy", affinities.detach().cpu().numpy())

        affinities = self.attention_dropout(affinities)
        # if attention_mask is None:
        #     np.save("../numpy_logs/torch_attention_dropout.npy", affinities.detach().cpu().numpy())

        affinities = affinities.view(B, self.heads_number, query_T, -1)
        value_vector = value_vector.permute(0, 2, 1, 3).view(B, self.heads_number, -1, self.embeddings_number)

        output = affinities @ value_vector  # [B, heads_num, T, head_size]
        output = output.view(B, self.heads_number, query_T, self.embeddings_number)
        output = output.transpose(1, 2)  # [B, T, heads_num, head_size]

        # if attention_mask is None:
        #     np.save("../numpy_logs/torch_attention_output.npy", output.detach().cpu().numpy())

        output = output.contiguous().view(B, query_T, self.embeddings_number * self.heads_number)  # [B, T, C]
        projection = self.output_projection(output)

        # if attention_mask is None:
        #     np.save("../numpy_logs/torch_output_projection.npy", projection.detach().cpu().numpy())

        output = self.output_dropout(self.output_projection(output))

        return output


class TransformerBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        em = kwargs["embeddings_number"]
        img_ch = kwargs["image_channels"]
        dropout_rate = kwargs["dropout_rate"]
        context_length = kwargs["context_length"]
        device = kwargs["device"]

        self.register_buffer(
            "triangle_mask",
            torch.tril(torch.ones(context_length, context_length, device=device)).view(
                1, 1, context_length, context_length
            ),
        )

        # Self attention
        self.self_attention = MultiHeadAttention(inputs_channels=(em, em, em), **kwargs)
        self.layer_normalization_1 = nn.LayerNorm(em, device=device)

        # Cross attention
        self.cross_attention = MultiHeadAttention(inputs_channels=(em, img_ch, img_ch), **kwargs)
        self.layer_normalization_2 = nn.LayerNorm(em, device=device)

        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(em, 4 * em, device=device),
            nn.ReLU(),
            nn.Linear(4 * em, em, device=device),
            nn.Dropout(dropout_rate),
        )
        self.layer_normalization_3 = nn.LayerNorm(em, device=device)
        self.last_attention_scores: Optional[torch.Tensor] = None

    def forward(self, image, caption):
        # Note: pre-norm formulation can be used
        # sa = self.self_attention(caption, caption, attention_mask=self.triangle_mask)
        # np.save("../numpy_logs/torch_self_attention.npy", sa.detach().cpu().numpy())

        x = torch.add(caption, self.self_attention(caption, caption, attention_mask=self.triangle_mask))
        x = self.layer_normalization_1(x)

        # mock_image = torch.ones((32, 49, 576)).to(torch.float).to("cuda")
        # mock_x = torch.ones((32, 20, 256)).to(torch.float).to("cuda")
        cross_attention = self.cross_attention(x, image)
        self.last_attention_scores = self.cross_attention.last_attention_scores
        # np.save("../numpy_logs/torch_cross_attention.npy", cross_attention.detach().cpu().numpy())
        # np.save(
        #     "../numpy_logs/torch_cross_attention_scores.npy",
        #     self.cross_attention.last_attention_scores.detach().cpu().numpy(),
        # )
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
