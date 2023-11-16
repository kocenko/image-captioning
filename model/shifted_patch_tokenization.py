import torch
from torch import nn
from torchvision.transforms.v2.functional import affine_image


class ShiftedPatchTokenizer(nn.Module):
    """Tokenizer used to transform an input image into a sequence of flattened patches

    Shifted Patch Tokenizer is used to tackle the problem of low receptive field of the encoder.
    When the image is split into non-overlapping patches, information about spacial relation between patches is lost.
    By grouping slightly shifted patches from the same region of an image, adjacent pixels are included in one vector.

    Args:
        image_size (tuple[int, int]): height and width of an image in pixels
        shift_pixels (tuple[int, int]): number of pixels to perform shift on (along height and width dimensions)
        patch_size (int): length of the patch side in pixels

    Attributes:
        image_size (tuple[int, int]): height and width of an image in pixels
        patch_size (int): length of the patch side in pixels
        shifts (list[int]): list of pairs of shift sizes used to perform four diagonal shifts
        flatten (nn.Flatten): layer used to flatten the output
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        shift_pixels: tuple[int, int],
        patch_size: int,
    ):
        super().__init__()
        assert (
            image_size[0] % patch_size == 0
        ), f"Image size should be divisible by patch size. Got image size: {image_size},  patch size: {patch_size}"

        self.image_size = image_size
        self.patch_size = patch_size
        x_shift, y_shift = shift_pixels

        # right-up, right-down, left-up, left-down
        self.shifts = [[x_shift, y_shift], [x_shift, -y_shift], [-x_shift, y_shift], [-x_shift, -y_shift]]
        self.flatten = nn.Flatten(start_dim=-4, end_dim=-1)

    def _shift_images(self, image: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                affine_image(image, angle=0.0, translate=shift, scale=1.0, shear=[0.0]).unsqueeze(1)
                for shift in self.shifts
            ]
            + [image.unsqueeze(1)],
            dim=1,
        )

    def _partition_images(self, image: torch.Tensor) -> torch.Tensor:
        # Expected input shape: [batches, shifts, channels, height, width]
        height, width = self.image_size[0], self.image_size[1]
        patches = torch.cat(
            [
                image[:, :, :, i: i + self.patch_size, j: j + self.patch_size].unsqueeze(1)
                for i in range(0, height, self.patch_size)
                for j in range(0, width, self.patch_size)
            ],
            dim=1,
        )
        return patches

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        shifted_images = self._shift_images(image)  # [batches, shifts, channels, height, width]
        patches = self._partition_images(shifted_images)  # [batches, patches, shifts, channels, patch_size, patch_size]
        flattened_patches = self.flatten(patches)  # [batches, patches, shifts*channels*patch_size*patch_size]
        return flattened_patches.to(torch.float)
