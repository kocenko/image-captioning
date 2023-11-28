from typing import Optional
import torch
from torchvision.transforms.v2 import Resize, Normalize, Compose, ToDtype
from torchvision.io import read_image

from data_processing.pretrained_models import PRETRAINED_MODELS


class ImageTransforms:
    def __init__(self, image_size: int, model_name: Optional[str] = None):
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.device = None

        if model_name:
            self.transformations = PRETRAINED_MODELS[model_name]["weights"].transforms(
                antialias=True, mean=tuple(self.mean), std=tuple(self.std)
            )
        else:
            self.transformations = Compose(
                [
                    Resize((image_size, image_size), antialias=True),
                    ToDtype(torch.float32, scale=True),
                    Normalize(mean=self.mean, std=self.std),
                ]
            )
        self.denormalize = Normalize(
            mean=[-single_mean / single_std for single_mean, single_std in zip(self.mean, self.std)],
            std=[1.0 / single_std for single_std in self.std],
        )

    def to(self, device: str):
        self.device = device

    def read_image(self, path_to_image: str) -> torch.Tensor:
        image = read_image(path_to_image)
        if self.device is not None:
            image = image.to(self.device)
        return image

    def denormalize(self, image: torch.Tensor) -> torch.Tensor:
        denormalized = self.denormalize(image)
        if self.device is not None:
            denormalized = denormalized.to(self.device)
        return denormalized

    def transform(self, image: torch.Tensor) -> torch.Tensor:
        transformed_image = self.transformations(image)
        if self.device is not None:
            transformed_image = transformed_image.to(self.device)
        return transformed_image
