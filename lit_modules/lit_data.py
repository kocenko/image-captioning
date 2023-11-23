from typing import Optional

from lightning import LightningDataModule
from torch.utils.data import DataLoader

from data_processing.custom_dataset import ImageCaptionDataset
from data_processing.tokenizer import Tokenizer
from data_processing.image_transforms import ImageTransforms
from data_processing.dataset_reader import load_flickr8k, load_flickr30k


class DataModule(LightningDataModule):
    DATA_READER_MAP = {
        "flickr8k": load_flickr8k,
        "flickr30k": load_flickr30k,
    }

    def __init__(
        self,
        dataset_name: str,
        dataset_paths: dict,
        max_sequence_size: int,
        vocabulary_size: int,
        image_size: int,
        batch_size: int,
    ):
        super().__init__()

        assert dataset_name in ["flickr8k", "flickr30k"], f"Dataset of given name {dataset_name} was not implemented"
        self.raw_datasets = DataModule.DATA_READER_MAP[dataset_name](**dataset_paths)
        self.tokenizer = Tokenizer([caption for _, caption in self.raw_datasets[0]], max_sequence_size, vocabulary_size)
        self.transform = ImageTransforms(image_size)
        self.batch_size = batch_size

        self.train: Optional[ImageCaptionDataset] = None
        self.val: Optional[ImageCaptionDataset] = None
        self.test: Optional[ImageCaptionDataset] = None

    def setup(self, stage: str) -> None:
        self.train = ImageCaptionDataset(self.raw_datasets[0], self.tokenizer, self.transform)
        self.val = ImageCaptionDataset(self.raw_datasets[1], self.tokenizer, self.transform)
        self.test = ImageCaptionDataset(self.raw_datasets[2], self.tokenizer, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)
