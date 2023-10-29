import os.path
import random
import tqdm
from typing import Any
from collections import defaultdict

from image_captioning.tokenizer import Tokenizer
from image_captioning.feature_extractor import FeatureExtractor

import torch
from torch.utils.data import Dataset


def load_flickr8k(
    tokens_path: str, train_path: str, valid_path: str, test_path: str, images_path: str
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    # Get the list of all images' paths
    images_paths = os.listdir(images_path)

    # Read file paths to every group and filter out non-existing paths
    train_paths, valid_paths, test_paths = [], [], []
    for path_to_file, array in zip([train_path, valid_path, test_path], [train_paths, valid_paths, test_paths]):
        with open(path_to_file, "r") as f:
            content = f.read()
        array.extend([path for path in content.splitlines() if path in images_paths])

    # Map paths to captions
    with open(tokens_path, "r") as f:
        image_caption_pair = [
            (line.split("\t", 1)[0].split("#", 1)[0], line.split("\t", 1)[1]) for line in f.read().splitlines()
        ]

    train = [
        (os.path.join(images_path, image_path), caption)
        for image_path, caption in image_caption_pair
        if image_path in train_paths
    ]
    valid = [
        (os.path.join(images_path, image_path), caption)
        for image_path, caption in image_caption_pair
        if image_path in valid_paths
    ]
    test = [
        (os.path.join(images_path, image_path), caption)
        for image_path, caption in image_caption_pair
        if image_path in test_paths
    ]

    return train, valid, test


class Sharder:
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
        """
        Splits list into even groups.

        Args:
            list_size (int): size of the input list
            elements_per_group (int): number of elements in each group

        Returns:
            Iterator for a list of indexes in each group
        """

        for i in range(0, list_size, elements_per_group):
            yield slice(i, i + elements_per_group)

    @staticmethod
    def __empty_directory(directory: str) -> None:
        """
        Deletes files from the list of paths

        Args:
            directory (str): path to the directory to be cleared
        """
        for file in [os.path.join(directory, file) for file in os.listdir(directory)]:
            os.remove(file)
            print(f"Deleted {file}")

    def load_and_transform_caption(self, captions: list[str]) -> list[torch.Tensor]:
        return [torch.tensor(self.tokenizer.encode(caption), device=self.device) for caption in captions]

    def load_and_transform_image(self, paths: list[str]) -> torch.Tensor:
        transformed = None
        for batch_slice in self.split_evenly(len(paths), self.batch_size):
            raw_images = [self.extractor.get_image_from_file(path).to(self.device) for path in paths[batch_slice]]
            transformed_batch = self.extractor.feed(torch.stack(raw_images, dim=0))

            if transformed is None:
                transformed = transformed_batch
            else:
                transformed = torch.cat((transformed, transformed_batch), dim=0)

        return transformed

    def save_shards(
        self, dataset: list[tuple[str, str]], shard_name: str, dest_folder: str, override: bool = False
    ) -> None:
        """
        Saves transformed dataset into shard files

        Args:
            dataset (list[tuple[str, str]]): list of pairs of image paths and captions
            shard_name (str): name of the shard, identifier (for example train, valid, test)
            dest_folder (str): path to the folder where shards are saved
            override (bool): whether to delete existing shards before creating new ones
        """

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        if override:
            self.__empty_directory(dest_folder)

        for i, shard in enumerate(tqdm.tqdm(self.split_evenly(len(dataset), self.shard_size))):
            image_paths = [img for img, _ in dataset[shard]]
            captions = [cap for _, cap in dataset[shard]]

            cap = self.load_and_transform_caption(captions)
            img = self.load_and_transform_image(image_paths)

            path_to_new_shard = os.path.join(dest_folder, f"{shard_name}_shard_{i}.pt")
            self.shards_names[shard_name].append(path_to_new_shard)
            torch.save((img, cap), path_to_new_shard)

    def load_shards(self, shard_folders: list[str], shard_names: list[str]) -> None:
        for folder, name in zip(shard_folders, shard_names):
            self.shards_names[name] = [os.path.join(folder, file) for file in os.listdir(folder)]


class ImageCaptionDataset(Dataset):
    """
    Custom class for dataset

    Attributes:
        img (torch.Tensor): tensor with extracted features from the images
        cap (list[str]): list with captions
        batches_indexes (list[int]): list of indexes of each batch
    """

    def __init__(self, shard_path: str, batch_size: int, device: Any) -> None:
        """
        Dataset initialization

        Args:
            shard_path (str): path to the shard file
        """

        self.img, self.cap = torch.load(shard_path, map_location=device)
        indexes = random.sample(range(len(self.cap)), len(self.cap))
        self.batches_indexes = [indexes[i : i + batch_size] for i in range(0, len(indexes), batch_size)]

    def __len__(self) -> int:
        """
        Returns:
             Length of the dataset
        """

        return len(self.cap)

    def __getitem__(self, item: int):
        """
        Fetches a batch of the given index

        Args:
             item (int): index of the sample to return

        Returns:
            A tuple with three elements: tensor of the transformed image and tokenized input and output (label) captions
        """

        batch_idx = self.batches_indexes[item]
        image_batch = torch.stack([self.img[i] for i in batch_idx])
        caption_batch = torch.nn.utils.rnn.pad_sequence([self.cap[i] for i in batch_idx], batch_first=True)
        return image_batch, caption_batch[:, :-1], caption_batch[:, 1:]


def custom_dataloader(split_name: str, sharder: Sharder, batch_size: int):
    shard_files = sharder.shards_names[split_name]

    for shard_file in shard_files:
        dataset = ImageCaptionDataset(shard_file, batch_size, sharder.device)
        for img, caption, label in dataset:
            yield img, caption, label


if __name__ == "__main__":
    pass
