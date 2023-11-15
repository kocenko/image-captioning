import torch
from torch import nn

from model.multihead_attention import MultiHeadAttention
from model.add_and_norm import ResidualLayerNormalization


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embeddings: int,
        dropout_rate: float,
        heads_num: int,
        patches_num: int,
        device: str,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            input_shapes=(embeddings, embeddings, embeddings),
            embeddings_number=embeddings,
            heads_number=heads_num,
            device=device,
            trainable_scale=True,
        )
        self.attention_dropout = nn.Dropout(dropout_rate)
        self.add_and_norm_1 = ResidualLayerNormalization(embeddings_number=embeddings, device=device)
        self.feed_forward = nn.Sequential(
            nn.Linear(embeddings, 4 * embeddings, device=device),
            nn.GELU(),
            nn.Linear(4 * embeddings, embeddings, device=device),
            nn.Dropout(dropout_rate),
        )
        self.ff_dropout = nn.Dropout(dropout_rate)
        self.add_and_norm_2 = ResidualLayerNormalization(embeddings_number=embeddings, device=device)

        # Used to ensure Locality Self Attention
        # noinspection PyTypeChecker
        self.register_buffer("diagonal_mask", torch.eye(patches_num, device=device) == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sa = self.self_attention(x, x, x, self.diagonal_mask)
        sa = self.attention_dropout(sa)
        x = self.add_and_norm_1(sa, x)
        ff = self.feed_forward(x)
        ff = self.ff_dropout(ff)
        x = self.add_and_norm_2(ff, x)
        return x
