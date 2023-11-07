import torch
from torch import nn

from model.add_and_norm import ResidualLayerNormalization
from model.multihead_attention import MultiHeadAttention


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embeddings: int,
        dropout_rate: float,
        heads_num: int,
        max_caption_length: int,
        device: str,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            input_shapes=(embeddings, embeddings, embeddings),
            embeddings_number=embeddings,
            heads_number=heads_num,
            dropout_rate=dropout_rate,
            device=device,
        )
        self.add_and_norm_1 = ResidualLayerNormalization(embeddings, device)

        self.cross_attention = MultiHeadAttention(
            input_shapes=(embeddings, embeddings, embeddings),
            embeddings_number=embeddings,
            heads_number=heads_num,
            dropout_rate=dropout_rate,
            device=device,
        )
        self.add_and_norm_2 = ResidualLayerNormalization(embeddings, device)

        self.feed_forward = nn.Sequential(
            nn.Linear(embeddings, 4 * embeddings, device=device),
            nn.GELU(),
            nn.Linear(4 * embeddings, embeddings, device=device),
            nn.Dropout(dropout_rate),
        )
        self.add_and_norm_3 = ResidualLayerNormalization(embeddings, device)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_caption_length, max_caption_length, device=device)).view(
                1, 1, max_caption_length, max_caption_length
            ),
        )

    def forward(self, image, caption):
        sa = self.self_attention(caption, caption, caption, self.causal_mask)
        x = self.add_and_norm_1(sa, caption)

        cr = self.cross_attention(x, image, image)
        x = self.add_and_norm_2(cr, x)

        ff = self.feed_forward(x)
        x = self.add_and_norm_3(ff, x)
        return x
