from typing import Optional

import torch
from torch import nn
from torchvision.transforms.v2.functional import affine_image


class ShiftImage(nn.Module):
    """Layer used to shift image

    Attributes:
        shift_pixels (list[int]): each element represents the shift in that axis, `expected shape: 2`
    """

    def __init__(self, shift_pixels: list[int]):
        super().__init__()
        self.shift_pixels = shift_pixels

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = affine_image(image, angle=0.0, translate=self.shift_pixels, scale=1.0, shear=[0.0])
        return x


class PatchTokenizer(nn.Module):
    """Tokenizer used to transform an input image into a sequence of flattened patches

    Shifted Patch Tokenizer is used to tackle the problem of low receptive field of the encoder.
    When the image is split into non-overlapping patches, information about spacial relation between patches is lost.
    By grouping slightly shifted patches from the same region of an image, adjacent pixels are included in one vector.

    Args:
        image_size (tuple[int, int]): height and width of an image in pixels
        patch_size (int): length of the patch side in pixels
        embeddings (int): dimension of a model
        device (str): device on which calculations are performed
        shift (Optional[int]): number of pixels to perform diagonal shift across

    Attributes:
        patch_embedding_layers (nn.ModuleList): list of convolutional layers with shifting layers
        flatten (nn.Flatten): layer used to flatten the output
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: int,
        embeddings: int,
        device: str,
        shift: Optional[int] = None
    ):
        super().__init__()
        assert (
            image_size[0] % patch_size == 0
        ), f"Image size should be divisible by patch size. Got image size: {image_size},  patch size: {patch_size}"

        shifted_patches = []
        if shift:
            # right-up, right-down, left-up, left-down
            shifts = [[shift, shift], [shift, -shift], [-shift, shift], [-shift, -shift]]
            shifted_patches.extend([ShiftImage(shift_pixels) for shift_pixels in shifts])

        self.patch_shifting_layers = nn.ModuleList(shifted_patches)
        self.projection = nn.Conv2d(
            3 * (5 if shift else 1), embeddings, kernel_size=patch_size, stride=patch_size, device=device
        )
        self.flatten = nn.Flatten(start_dim=2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x = torch.cat([image] + [layer(image) for layer in self.patch_shifting_layers], dim=1)
        x = self.projection(x)
        x = self.flatten(x)
        x = x.transpose(1, 2)
        return x
