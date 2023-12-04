import torch
from torch import nn

from model.patch_tokenization import PatchTokenizer
from model.transformer_sublayers import LocalitySelfAttention, FeedForward


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
        self.positional_embedding = nn.Embedding(patches_num, embeddings)
        self.patch_tokenizer = PatchTokenizer(image_size, patch_size, embeddings, None)
        self.register_buffer("sequence_indices", torch.arange(patches_num), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = self.patch_tokenizer(image)
        positional_embedding = self.positional_embedding(self.sequence_indices)
        output = patches + positional_embedding
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
        self.self_attention = LocalitySelfAttention(embeddings, heads_num, dropout_rate)
        self.feed_forward = FeedForward(embeddings, 4, dropout_rate, False)

        # noinspection PyTypeChecker
        self.register_buffer("diagonal_mask", torch.eye(patches_num) == 1, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.self_attention(x)
        x = self.feed_forward(x)
        return x
