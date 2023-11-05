import os.path
import random
import tqdm
from typing import Any
from collections import defaultdict

import torch
from torch.utils.data import Dataset

from data_processing.tokenizer import Tokenizer
from data_processing.feature_extractor import FeatureExtractor

class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        dataset: list[tuple[str, str]],
        tokenizer: Tokenizer,
        extractor: FeatureExtractor,
        device: str = 'cpu'
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.device: Any = torch.device(device)

    def collate(self, batch):
        images, captions, labels = zip(*batch)
        images_batch = torch.stack(images)
        images_features = self.extractor.feed(images_batch)
        captions = torch.tensor(captions, device=self.device)
        labels = torch.tensor(labels, device=self.device)
        return images_features, captions, labels

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index: int) -> Any:
        img = self.extractor.get_image_from_file(self.dataset[index][0]).to(self.device)
        cap = self.tokenizer.encode(self.dataset[index][1])
        return img, cap[:-1], cap[1:]

class DataCachingManager:
    def __init__(
        self,
        tokenizer: Tokenizer,
        extractor: FeatureExtractor,
        shard_size: int,
        batch_size: int,
        device: str = "cpu",
    ) -> None:
        self.tokenizer = tokenizer
        self.extractor = extractor
        self.shard_size = shard_size
        self.batch_size = batch_size
        self.shards_names = defaultdict(list)
        self.device: Any = torch.device(device)

    @staticmethod
    def split_evenly(list_size: int, elements_per_group: int) -> list[Any]:
        """Splits list into even groups.

        Args:
            list_size (int): size of the input list
            elements_per_group (int): number of elements in each group

        Returns:
            Iterator for a list of indexes in each group
        """

        for i in range(0, list_size, elements_per_group):
            yield slice(i, i + elements_per_group)


    def __load_and_transform_caption(self, captions: list[str]) -> list[torch.Tensor]:
        return [torch.tensor(self.tokenizer.encode(caption), device=self.device) for caption in captions]

    def __load_and_transform_image(self, paths: list[str]) -> torch.Tensor:
        """Extracts features in batches and stacks the results"""
        
        transformed = []
        for batch_slice in self.split_evenly(len(paths), self.batch_size):
            raw_images = [self.extractor.get_image_from_file(path).to(self.device) for path in paths[batch_slice]]
            transformed.append(self.extractor.feed(torch.stack(raw_images, dim=0)))
        transformed = torch.cat(transformed, dim=0)
        return transformed

    def save_shards(self, dataset: list[tuple[str, str]], shard_name: str, dest_folder: str) -> None:
        """Saves transformed dataset into shard files

        Args:
            dataset (list[tuple[str, str]]): list of pairs of image paths and captions
            shard_name (str): name of the shard, identifier (for example train, valid, test)
            dest_folder (str): path to the folder where shards are saved
        """

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        for i, shard in enumerate(tqdm.tqdm(self.split_evenly(len(dataset), self.shard_size))):
            image_paths = [img for img, _ in dataset[shard]]
            captions = [cap for _, cap in dataset[shard]]

            cap = self.__load_and_transform_caption(captions)
            img = self.__load_and_transform_image(image_paths)

            path_to_new_shard = os.path.join(dest_folder, f"{shard_name}_shard_{i}.pt")
            self.shards_names[shard_name].append(path_to_new_shard)
            torch.save((img, cap), path_to_new_shard)

    def load_shards(self, shard_folders: list[str], shard_names: list[str]) -> None:
        for folder, name in zip(shard_folders, shard_names):
            self.shards_names[name] = [os.path.join(folder, file) for file in os.listdir(folder)]


class CachedDataset(Dataset):
    """Custom class for dataset

    Attributes:
        img (torch.Tensor): tensor with extracted features from the images
        cap (list[str]): list with captions
        batches_indexes (list[int]): list of indexes of each batch
    """

    def __init__(self, shard_path: str, batch_size: int, device: Any) -> None:
        """Dataset initialization

        Args:
            shard_path (str): path to the shard file
        """

        self.img, self.cap = torch.load(shard_path, map_location=device)
        indexes = random.sample(range(len(self.cap)), len(self.cap))
        self.batches_indexes = [indexes[i: i + batch_size] for i in range(0, len(indexes), batch_size)]

    def __len__(self) -> int:
        return len(self.cap)

    def __getitem__(self, item: int):
        """Fetches a batch of the given index

        Args:
             item (int): index of the sample to return

        Returns:
            A tuple with three elements: tensor of the transformed image and tokenized input and output (label) captions
        """

        batch_idx = self.batches_indexes[item]
        image_batch = torch.stack([self.img[i] for i in batch_idx])
        caption_batch = torch.stack([self.cap[i] for i in batch_idx])
        return image_batch, caption_batch[:, :-1], caption_batch[:, 1:]


def custom_dataloader(split_name: str, sharder: DataCachingManager, batch_size: int):
    shard_files = sharder.shards_names[split_name]
    random.shuffle(shard_files)

    for shard_file in shard_files:
        dataset = CachedDataset(shard_file, batch_size, sharder.device)
        for img, caption, label in dataset:
            yield img, caption, label


if __name__ == "__main__":
    pass
