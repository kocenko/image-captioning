from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.flattener = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        # Expected input (B, C, H, W)
        x = self.flattener(x)
        x = x.transpose(-2, -1)
        return x


class SingleHeadAttention(nn.Module):
    def __init__(self, input_shapes: Tuple[int, int, int], mask_out: bool = True, **kwargs):
        super().__init__()
        context_length = kwargs["context_length"]
        dropout_rate = kwargs["dropout_rate"]
        head_size = kwargs["head_size"]
        device = kwargs["device"]

        query_input_shape, key_input_shape, value_input_shape = input_shapes

        self.mask_out = mask_out
        self.queries_weights = nn.Linear(query_input_shape, head_size, bias=False, device=device)
        self.keys_weights = nn.Linear(key_input_shape, head_size, bias=False, device=device)
        self.values_weights = nn.Linear(value_input_shape, head_size, bias=False, device=device)
        self.masking_triangle = torch.tril(torch.ones(context_length, context_length, device=device))
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sample):
        query, key_or_value = sample

        _, key_sequence_shape, key_channels_shape = key_or_value.shape

        query_vector = self.queries_weights(query)
        key_vector = self.keys_weights(key_or_value)
        value_vector = self.values_weights(key_or_value)

        affinities = query_vector @ key_vector.transpose(-2, -1)  # Transposing channels with sequence
        affinities *= key_channels_shape**(-.5)
        if self.mask_out:
            S = key_sequence_shape
            affinities = affinities.masked_fill(self.masking_triangle[:S, :S] == 0, float('-inf'))
        affinities = F.softmax(affinities, dim=-1)
        affinities = self.dropout(affinities)

        return affinities @ value_vector


class MultiHeadAttention(nn.Module):
    def __init__(self, input_shapes: Tuple[int, int, int], mask_out: bool = False, **kwargs):
        super().__init__()
        heads_number = kwargs["heads_number"]
        embeddings_number = kwargs["embeddings_number"]
        dropout_rate = kwargs["dropout_rate"]
        device = kwargs["device"]

        _, _, value_shape = input_shapes

        self.heads = nn.ModuleList([SingleHeadAttention(input_shapes, mask_out, **kwargs) for _ in range(heads_number)])
        self.projection = nn.Linear(value_shape, embeddings_number, device=device)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, sample):
        query, key_or_value = sample
        x = torch.cat([single_head((query, key_or_value)) for single_head in self.heads], dim=-1)
        x = self.projection(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        embeddings_number = kwargs["embeddings_number"]
        dropout_rate = kwargs["dropout_rate"]
        image_size = kwargs["image_size"]
        device = kwargs["device"]

        self_input_shapes = (embeddings_number, embeddings_number, embeddings_number)  # Q, K, V
        cross_input_shapes = (embeddings_number, image_size, image_size)  # Q, K, V

        # Self attention
        self.layer_normalization_1 = nn.LayerNorm(embeddings_number, device=device)
        self.self_attention = MultiHeadAttention(self_input_shapes, mask_out=True, **kwargs)

        # Cross attention
        self.layer_normalization_2 = nn.LayerNorm(embeddings_number, device=device)
        self.cross_attention = MultiHeadAttention(cross_input_shapes, **kwargs)

        # Feed forward
        self.layer_normalization_3 = nn.LayerNorm(embeddings_number, device=device)
        self.feed_forward = nn.Sequential(nn.Linear(embeddings_number, 4 * embeddings_number, device=device),
                                          nn.ReLU(),
                                          nn.Linear(4 * embeddings_number, embeddings_number, device=device),
                                          nn.Dropout(dropout_rate))

    def forward(self, sample):
        image, caption = sample
        x = self.layer_normalization_1(caption)
        x = x + self.self_attention((x, x))
        x = self.layer_normalization_2(x)
        x = x + self.cross_attention((x, image))
        x = self.layer_normalization_3(x)
        x = x + self.feed_forward(x)

        return x


class TokenEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        embeddings_number = kwargs["embeddings_number"]
        vocabulary_size = kwargs["vocabulary_size"]
        context_length = kwargs["context_length"]
        self.device = kwargs["device"]

        self.token_embedding_table = nn.Embedding(vocabulary_size, embeddings_number, device=self.device)
        self.positional_embedding = nn.Embedding(context_length, embeddings_number, device=self.device)

    @staticmethod
    def __positional_embedding(batch_size: int, embedding_size: int) -> torch.Tensor:
        # Fixed positional embedding
        raise NotImplementedError

    def forward(self, sequence):
        _, sequence_size = sequence.shape
        token_embedding = self.token_embedding_table(sequence)
        positional_embedding = self.positional_embedding(torch.arange(sequence_size, device=self.device).unsqueeze(0))
        sequence = token_embedding + positional_embedding
        return sequence


class Decoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        embeddings_number = kwargs["embeddings_number"]
        vocabulary_size = kwargs["vocabulary_size"]
        blocks_number = kwargs["blocks_number"]
        device = kwargs["device"]
        self.device = device

        # Embeddings (with positional)
        self.image_flattener = EncoderBlock()
        self.embedding = TokenEmbedding(**kwargs)
        self.linear = nn.Linear(embeddings_number, vocabulary_size, device=device)
        self.blocks = nn.Sequential(*[TransformerBlock(**kwargs) for _ in range(blocks_number)],
                                    nn.LayerNorm(embeddings_number, device=device))
        self.layer_normalization = nn.LayerNorm(embeddings_number, device=device)

    def forward(self, image, caption, targets=None):
        caption = self.embedding(caption)
        image = self.image_flattener(image)
        x = self.blocks((image, caption))
        x = self.layer_normalization(x)
        logits = self.linear(x)

        return logits
