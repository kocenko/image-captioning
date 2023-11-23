from typing import Any

import torch
from torch.utils.data import Dataset

from data_processing.image_transforms import ImageTransforms
from data_processing.tokenizer import Tokenizer


class ImageCaptionDataset(Dataset):
    """Custom dataset derived from the PyTorch dataset class

    Args:
        dataset (list[tuple[str, str]]): list of tuple pairs: (image path, raw caption)
        tokenizer (Tokenizer): tokenizer object
        transform (ImageTransforms): transforms image

    Attributes:
        dataset (list[tuple[str, str]]): list of tuple pairs: (image path, raw caption)
        tokenizer (Tokenizer): tokenizer object
        transform (ImageTransforms): transforms image
    """

    def __init__(
        self,
        dataset: list[tuple[str, str]],
        tokenizer: Tokenizer,
        transform: ImageTransforms,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        img = self.transform.read_image(self.dataset[index][0])
        img = self.transform.transform(img)
        cap = torch.tensor(self.tokenizer.encode(self.dataset[index][1]))
        return img, cap[:-1], cap[1:]


if __name__ == "__main__":
    pass
