from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms.v2 import Resize


from data_processing.tokenizer import Tokenizer


class ImageCaptionDataset(Dataset):
    """Custom dataset derived from the PyTorch dataset class

    Args:
        dataset (list[tuple[str, str]]): list of tuple pairs: (image path, raw caption)
        tokenizer (Tokenizer): tokenizer object
        device (str): indicates on which device the image will be saved

    Attributes:
        dataset (list[tuple[str, str]]): list of tuple pairs: (image path, raw caption)
        tokenizer (Tokenizer): tokenizer object
        device (str): indicates on which device the image will be saved
    """
    def __init__(
        self,
        dataset: list[tuple[str, str]],
        image_size: tuple[int, int],
        tokenizer: Tokenizer,
        device: str,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device
        self.resize = Resize(image_size, antialias=True)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Any:
        img = read_image(self.dataset[index][0]).to(self.device)
        img = self.resize(img)
        cap = torch.tensor(self.tokenizer.encode(self.dataset[index][1]), device=self.device)
        return img, cap[:-1], cap[1:]


if __name__ == "__main__":
    pass
