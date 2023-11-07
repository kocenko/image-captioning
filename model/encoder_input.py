import torch
from torch import nn

from model.shifted_patch_tokenization import ShiftedPatchTokenizer


class EncoderInput(nn.Module):
    def __init__(
        self,
        image_size: tuple[int, int],
        shift_pixels: tuple[int, int],
        patch_size: int,
        embeddings: int,
        device: str,
    ):
        super().__init__()
        features_num = 4 * 3 * image_size[0] * image_size[1]  # shifts * channels * height * width
        patches_num = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        self.device = device
        self.patch_tokenizer = ShiftedPatchTokenizer(image_size, shift_pixels, patch_size)
        self.positional_embedding = nn.Embedding(patches_num, embeddings, device=device)
        self.patch_embedding = nn.Embedding(features_num, embeddings, device=device)  # No padding performed

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        patches = self.patch_tokenizer(image)
        patch_embedding = self.patch_embedding(patches)
        positional_embedding = self.positional_embedding(torch.arange(patches.shape[1], device=self.device))
        return patch_embedding + positional_embedding
