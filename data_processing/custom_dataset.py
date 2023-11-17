from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2 import Resize, Normalize

from data_processing.image_transforms import ImageTransforms
from data_processing.tokenizer import Tokenizer


class ImageCaptionDataset(Dataset):
    """Custom dataset derived from the PyTorch dataset class

    Args:
        dataset (list[tuple[str, str]]): list of tuple pairs: (image path, raw caption)
        tokenizer (Tokenizer): tokenizer object
        transform (ImageTransforms): transforms image
        device (str): indicates on which device the image will be saved

    Attributes:
        dataset (list[tuple[str, str]]): list of tuple pairs: (image path, raw caption)
        tokenizer (Tokenizer): tokenizer object
        transform (ImageTransforms): transforms image
        device (str): indicates on which device the image will be saved
    """

    def __init__(
        self,
        dataset: list[tuple[str, str]],
        tokenizer: Tokenizer,
        transform: ImageTransforms,
        device: str,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transform
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        img = self.transform.read_image(self.dataset[index][0]).to(self.device)
        img = self.transform.transform(img)
        cap = torch.tensor(self.tokenizer.encode(self.dataset[index][1]), device=self.device)
        return img, cap[:-1], cap[1:]


if __name__ == "__main__":
    pass
