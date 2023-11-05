import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        input_shapes: tuple[int, int, int],
        embeddings_number: int,
        heads_number: int,
        dropout_rate: float,
        device: str,
        **kwargs
    ) -> None:
        super().__init__()

        self.device = device
        self.num_heads = heads_number
        self.key_dim = embeddings_number
        self.query_projection = nn.Linear(input_shapes[0], embeddings_number*heads_number, bias=True, device=device)
        self.key_projection = nn.Linear(input_shapes[1], embeddings_number*heads_number, bias=True, device=device)
        self.value_projection = nn.Linear(input_shapes[2], embeddings_number*heads_number, bias=True, device=device)
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=-1)
        self.output_projection = nn.Linear(embeddings_number*heads_number, embeddings_number, bias=True, device=device)
        self.output_dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch. Tensor,
        mask: bool,
    ) -> torch.Tensor:
        
        type_of_att = 'self' if mask else 'cross'


        # Remembering input dimensions
        B, T_q, C_q = query.shape
        _, T_k, C_k = key.shape

        # Projecting inputs
        # In this case key_dim = value_dim = embeddings. Also, T_k = T_v
        query = self.query_projection(query)  # [B, T_q, key_dim * num_heads]
        key   = self.key_projection(key)      # [B, T_k, key_dim * num_heads]
        value = self.value_projection(value)  # [B, T_v, val_dim * num_heads]

        # Splitting heads
        query = query.view(B, T_q, self.num_heads, self.key_dim).transpose(1, 2)  # [B, num_heads, T_q, key_dim]
        key   = key.view(B, T_k, self.num_heads, self.key_dim).transpose(1, 2)    # [B, num_heads, T_k, key_dim]
        value = value.view(B, T_k, self.num_heads, self.key_dim).transpose(1, 2)  # [B, num_heads, T_v, val_dim]

        # np.save(f'../numpy_logs/torch_key_{type_of_att}.npy', key.transpose(1, 2).detach().cpu().numpy())
        # np.save(f'../numpy_logs/torch_value_{type_of_att}.npy', value.transpose(1, 2).detach().cpu().numpy())


        # Scaling query
        query /= math.sqrt(self.key_dim)
        # np.save(f'../numpy_logs/torch_query_{type_of_att}.npy', query.transpose(1, 2).detach().cpu().numpy())


        # Dot-product between the query and the key
        affinity = query @ key.transpose(-2, -1)  # [B, num_heads, T_q, T_k]
        
        # np.save(f'../numpy_logs/torch_affinity_{type_of_att}.npy', affinity.detach().cpu().numpy())

        # Masked softmax
        if mask:
            causal_mask = torch.tril(torch.ones(T_q, T_k, device=self.device)).view(1, 1, T_q, T_k)
            affinity += (causal_mask == 0) * -1e9
        # np.save(f'../numpy_logs/torch_masked_{type_of_att}.npy', affinity.detach().cpu().numpy())
        affinity = self.softmax(affinity)
        # np.save(f'../numpy_logs/torch_softmax_{type_of_att}.npy', affinity.detach().cpu().numpy())
        affinity = self.attention_dropout(affinity)
        # np.save(f'../numpy_logs/torch_dropout_{type_of_att}.npy', affinity.detach().cpu().numpy())


        # # Output score
        # score = affinity @ value  # [B, num_heads, T_v, val_dim]
        # score = score.transpose(1, 2)
        score = torch.einsum('acbe,aecd->abcd', affinity, value.transpose(1, 2))
        # np.save(f'../numpy_logs/torch_out_{type_of_att}.npy', score.detach().cpu().numpy())

        # Output projection
        score = score.reshape(B, T_q, -1)
        score = self.output_projection(score)
        # np.save(f'../numpy_logs/torch_projection_{type_of_att}.npy', score.detach().cpu().numpy())

        score = self.output_dropout(score)
        return score


class TransformerBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        embeddings = kwargs["embeddings_number"]
        channels = kwargs["image_channels"]
        dropout_rate = kwargs["dropout_rate"]
        context_length = kwargs["context_length"]
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
        sa = self.self_attention(caption, caption, caption, True)
        # np.save('../numpy_logs/torch_self_att.npy', sa.detach().cpu().numpy())
        x = torch.add(caption, sa)
        # np.save('../numpy_logs/torch_self_add.npy', x.detach().cpu().numpy())
        x = self.layer_normalization_1(x)
        # np.save('../numpy_logs/torch_self_norm.npy', x.detach().cpu().numpy())

        cr = self.cross_attention(x, image, image, False)
        # np.save('../numpy_logs/torch_cross_att.npy', cr.detach().cpu().numpy())
        x = torch.add(x, cr)
        # np.save('../numpy_logs/torch_cross_add.npy', x.detach().cpu().numpy())
        x = self.layer_normalization_2(x)
        # np.save('../numpy_logs/torch_cross_norm.npy', x.detach().cpu().numpy())

        x = x + self.feed_forward(x)
        # np.save('../numpy_logs/torch_ff_add.npy', x.detach().cpu().numpy())

        x = self.layer_normalization_3(x)
        # np.save('../numpy_logs/torch_ff_norm.npy', x.detach().cpu().numpy())

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
