import torch
from torch import nn

from model.multihead_attention import MultiHeadAttention
from model.patch_tokenization import PatchTokenizer


class EncoderInput(nn.Module):
    def __init__(
        self,
        image_size: int,
        shift_pixels: int,
        patch_size: int,
        embeddings: int,
    ):
        super().__init__()
        patches_num = (image_size // patch_size) ** 2
        self.patch_tokenizer = PatchTokenizer(image_size, patch_size, embeddings, shift_pixels)
        self.positional_embedding = nn.Embedding(patches_num, embeddings)
        self.register_buffer("sequence_indices", torch.arange(patches_num), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = self.patch_tokenizer(image)
        positional_embedding = self.positional_embedding(self.sequence_indices)
        output = patches + positional_embedding
        # TODO: Consider dropout here
        return output


class EncoderBlock(nn.Module):
    def __init__(
        self,
        embeddings: int,
        dropout_rate: float,
        heads_num: int,
        patches_num: int,
    ):
        super().__init__()
        self.pre_normalization = nn.LayerNorm(embeddings)
        self.attention = MultiHeadAttention(
            input_shapes=(embeddings, embeddings, embeddings),
            embeddings_number=embeddings,
            heads_number=heads_num,
            trainable_scale=True,
        )
        self.post_normalization = nn.LayerNorm(embeddings)

        self.feed_forward = nn.Sequential(
            nn.Linear(embeddings, 4 * embeddings),
            nn.GELU(),
            nn.Linear(4 * embeddings, embeddings),
            nn.Dropout(dropout_rate),
        )

        # Used to ensure Locality Self Attention
        # noinspection PyTypeChecker
        self.register_buffer("diagonal_mask", torch.eye(patches_num) == 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.pre_normalization(x)
        sa = self.attention(x_norm, x_norm, x_norm, self.diagonal_mask)
        x = x + sa
        x_post_norm = self.post_normalization(x)
        ff = self.feed_forward(x_post_norm)
        x = x + ff
        return x
