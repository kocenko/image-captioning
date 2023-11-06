from typing import Any

from torch.utils.data import Dataset
from torchvision.io import read_image

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
        tokenizer: Tokenizer,
        device: str,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Any:
        img = read_image(self.dataset[index][0]).to(self.device)
        cap = self.tokenizer.encode(self.dataset[index][1])
        return img, cap[:-1], cap[1:]


if __name__ == "__main__":
    pass
