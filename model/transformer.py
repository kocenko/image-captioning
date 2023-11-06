import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

from model.multihead_attention import MultiHeadAttention
from model.add_and_norm import ResidualLayerNormalization


class TransformerBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        embeddings = kwargs["embeddings_number"]
        channels = kwargs["image_channels"]
        dropout_rate = kwargs["dropout_rate"]
        context_length = kwargs["context_length"]
        heads_number = kwargs["heads_number"]
        device = kwargs["device"]

        # Self attention
        self.self_attention = MultiHeadAttention(
            input_shapes=(embeddings, embeddings, embeddings),
            embeddings_number=embeddings,
            heads_number=heads_number,
            dropout_rate=dropout_rate,
            device=device,
        )
        self.add_and_norm_1 = ResidualLayerNormalization(embeddings, device)

        # Cross attention
        self.cross_attention = MultiHeadAttention(
            input_shapes=(embeddings, channels, channels),
            embeddings_number=embeddings,
            heads_number=heads_number,
            dropout_rate=dropout_rate,
            device=device,
        )
        self.add_and_norm_2 = ResidualLayerNormalization(embeddings, device)

        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(embeddings, 2 * embeddings, device=device),
            nn.ReLU(),
            nn.Linear(2 * embeddings, embeddings, device=device),
            nn.Dropout(dropout_rate),
        )
        self.add_and_norm_3 = ResidualLayerNormalization(embeddings. device)

    def forward(self, image, caption):
        sa = self.self_attention(caption, caption, caption, True)
        x = self.add_and_norm_1(sa, caption)

        cr = self.cross_attention(x, image, image, False)
        x = self.add_and_norm_2(cr, x)

        ff = self.feed_forward(x)
        x = self.add_and_norm_3(ff, x)
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
        # np.save('../numpy_logs/torch_image.npy', image.detach().cpu().numpy())

        x = self.embedding(caption)
        # np.save('../numpy_logs/torch_embedding.npy', x.detach().cpu().numpy())


        for block in self.blocks:
            x = block(image, x)

        logits = self.output_layer(x)
        # np.save('../numpy_logs/torch_output_layer.npy', logits.detach().cpu().numpy())

        return logits
