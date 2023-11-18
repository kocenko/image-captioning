import torch
from torch import nn

from model.multihead_attention import MultiHeadAttention
from model.add_and_norm import ResidualLayerNormalization
from model.patch_tokenization import PatchTokenizer


class EncoderInput(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        shift_pixels: int,
        patch_size: int,
        embeddings: int,
        device: str,
    ):
        super().__init__()
        patches_num = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.patch_tokenizer = PatchTokenizer(image_size, patch_size, embeddings, device)
        self.positional_embedding = nn.Embedding(patches_num, embeddings, device=device)
        self.layer_normalization = nn.LayerNorm(embeddings, device=device)
        self.register_buffer("sequence_indices", torch.arange(patches_num, device=device), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = self.patch_tokenizer(image)
        patch_embedding = self.layer_normalization(patches)
        positional_embedding = self.positional_embedding(self.sequence_indices)
        return patch_embedding + positional_embedding


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
