import torch
from torch import nn
from torchvision.transforms.v2 import Resize
from torchvision.transforms.v2.functional import affine_image


class ShiftedPatchTokenizer(nn.Module):
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

        self.resize = Resize(image_size, antialias=True)
        self.flatten = nn.Flatten(start_dim=-4, end_dim=-1)

    def _shift_images(self, image: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [
                affine_image(image, angle=0.0, translate=shift, scale=1.0, shear=[0.0]).unsqueeze(0)
                for shift in self.shifts
            ],
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
        resized_image = self.resize(image)  # [batches, channels, height, width]
        shifted_images = self._shift_images(resized_image)  # [batches, shifts, channels, height, width]
        patches = self._partition_images(shifted_images)  # [batches, patches, shifts, channels, height, width]
        flattened_patches = self.flatten(patches)  # [batches, patches, shifts*channels*height*width]
        return flattened_patches
