import torch
from torchvision.transforms.v2 import Resize, Normalize, Compose, ToDtype
from torchvision.io import read_image


class ImageTransforms:
    IMAGENET_MEAN = [0.5, 0.5, 0.5]
    IMAGENET_STD = [0.5, 0.5, 0.5]

    def __init__(self, image_size: tuple[int, int]):
        resize = Resize(image_size, antialias=True)
        normalize = Normalize(mean=ImageTransforms.IMAGENET_MEAN, std=ImageTransforms.IMAGENET_STD)
        change_data_type = ToDtype(torch.float32, scale=True)
        self.transformations = Compose([resize, change_data_type, normalize])

    @staticmethod
    def read_image(path_to_image: str) -> torch.Tensor:
        return read_image(path_to_image)

    def transform(self, image: torch.Tensor) -> torch.Tensor:
        transformed_image = self.transformations(image)
        return transformed_image
