from typing import Optional, Union
import torch
from torchvision.transforms.v2 import Resize, Normalize, Compose, ToDtype
from torchvision.io import read_image

from data_processing.pretrained_models import PRETRAINED_MODELS


class ImageTransforms:
    def __init__(self, model_name: Optional[str] = None):
        model_config = PRETRAINED_MODELS[model_name]
        self.model_name = model_name
        self.mean = model_config['mean']
        self.std = model_config['std']
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transformations = model_config['transforms']
        self.denormalize = Normalize(
            mean=[-single_mean / single_std for single_mean, single_std in zip(self.mean, self.std)],
            std=[1.0 / single_std for single_std in self.std],
        )

    def read_image(self, path_to_image: str) -> torch.Tensor:
        image = read_image(path_to_image).to(self.device)
        return image

    def denormalize_image(self, image: torch.Tensor) -> torch.Tensor:
        denormalized = self.denormalize(image).to(self.device)
        return denormalized

    def transform(self, image: torch.Tensor) -> Union[torch.Tensor, dict[str, torch.Tensor]]:
        if self.model_name == 'vit':
            transformed = self.transformations(images=image, return_tensors='pt')['pixel_values']
            return transformed
        transformed = self.transformations(image).to(self.device)
        return transformed
