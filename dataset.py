import os.path
import random
from typing import Any

from tokenizer import Tokenizer
from feature_extractor import FeatureExtractor

import torch
from torch.utils.data import Dataset, DataLoader


class Sharder:
    def __init__(
            self,
            tokenizer: Tokenizer,
            extractor: FeatureExtractor,
            shard_size: int,
            batch_size: int,
            split_ratio: tuple[int, int, int] = (.7, .2, .1),
            device: str = "cpu"
    ) -> None:

        self.tokenizer = tokenizer
        self.extractor = extractor
        self.shard_size = shard_size
        self.batch_size = batch_size
        self.dataset_size = len(tokenizer.image_paths)
        self.split_ratio = split_ratio
        self.device: Any = torch.device(device)
        self.split_indexes = {}

        self.set_split_indexes()

        # Create directories
        self.shard_folders = ['shards/train', 'shards/valid', 'shards/test']
        for split in self.shard_folders:
            if not os.path.exists(split):
                os.makedirs(split)

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

    def set_split_indexes(self) -> None:
        train_thresh, valid_thresh = self.split_ratio[0], self.split_ratio[0] + self.split_ratio[1],
        random_indexes = random.sample(range(self.dataset_size), self.dataset_size)

        self.split_indexes = {
            'train': random_indexes[: int(train_thresh * self.dataset_size)],
            'valid': random_indexes[int(train_thresh * self.dataset_size): int(valid_thresh * self. dataset_size)],
            'test': random_indexes[int(valid_thresh * self.dataset_size):],
        }

        for split_name, index_array in self.split_indexes.items():
            self.split_indexes[split_name] = [
                index_array[ids] for ids in self.split_evenly(len(index_array), self.shard_size)
            ]

    def __empty_directory(self) -> None:
        """
        Deletes files from the list of paths
        """
        for split in self.shard_folders:
            for file in [os.path.join(split, file) for file in os.listdir(split)]:
                os.remove(file)
                print(f"Deleted {file}")

    def load_and_transform_caption(self, captions: list[str]) -> torch.Tensor:
        tokenized_captions = [self.tokenizer.encode(caption) for caption in captions]
        tokenized_caption = torch.tensor(tokenized_captions, device=self.device)
        return tokenized_caption

    def load_and_transform_image(self, paths: list[str]) -> torch.Tensor:
        transformed = None
        for batch_slice in self.split_evenly(len(paths), self.batch_size):
            raw_images = [self.extractor.get_image_from_file(path).to(self.device) for path in paths[batch_slice]]
            transformed_batch = self.extractor.feed(torch.stack(raw_images, dim=0))

            if transformed is None:
                transformed = transformed_batch
                print(f"Shape of the features batch: {transformed_batch.shape}")
            else:
                transformed = torch.cat((transformed, transformed_batch), dim=0)

        return transformed

    def save_shards(self, override: bool = False) -> None:
        """
        Saves transformed dataset into shard files

        Args:
            override (bool): whether to delete existing shards before creating new ones
        """

        if override:
            self.__empty_directory()

        for i, key in enumerate(self.split_indexes):
            for j, shard in enumerate(self.split_indexes[key]):
                image_paths = [self.tokenizer.image_paths[i] for i in shard]
                captions = [self.tokenizer.captions[i] for i in shard]

                img = self.load_and_transform_image(image_paths)
                cap = self.load_and_transform_caption(captions)

                torch.save((img, cap), os.path.join(self.shard_folders[i], f"{key}_shard_{j}.pt"))
                print(f"Saved: {key}_shard_{j}")


class ImageCaptionDataset(Dataset):
    """
    Custom class for dataset

    Attributes:
        image_features (torch.Tensor): tensor with extracted features from the images
        captions (torch.Tensor): tensor with input captions (tokenized)
        labels (torch.Tensor): tensor with expected captions (tokenized)
    """

    def __init__(self, shard_path: str, device: Any) -> None:
        """
        Dataset initialization

        Args:
            shard_path (str): path to the shard file
        """

        img, cap = torch.load(shard_path, map_location=device)
        self.image_features = img
        self.captions = cap[..., :-1]
        self.labels = cap[..., 1:]

    def __len__(self) -> int:
        """
        Returns:
             Length of the dataset
        """

        return len(self.captions)

    def __getitem__(self, item: int):
        """
        Fetches a sample of the given index

        Args:
             item (int): index of the sample to return

        Returns:
            A tuple with three elements: tensor of the transformed image and tokenized input and output (label) captions
        """

        transformed_image = self.image_features[item]
        input_caption = self.captions[item]
        label_caption = self.labels[item]
        return transformed_image, input_caption, label_caption


def custom_dataloader(split_name: str, sharder: Sharder, batch_size: int):
    mapping = {'train': 0, 'valid': 1, 'test': 2}
    shard_folder = sharder.shard_folders[mapping[split_name]]
    shard_files = [file for file in os.listdir(shard_folder)]

    for shard_file in shard_files:
        dataset = ImageCaptionDataset(os.path.join(shard_folder, shard_file), sharder.device)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for (img, caption, label) in dataloader:
            yield img, caption, label


if __name__ == "__main__":
    file_path = 'dataset/captions.txt'
    folder = 'dataset/images/'

    with open(file_path, "r") as f:
        raw_file = f.read()

    fe = FeatureExtractor(model_name="mnasnet0_75", device="cpu")
    fe.slice_net("layers.15", overwrite_model=True)
    tk = Tokenizer(raw_file, folder)
    sh = Sharder(tk, fe, batch_size=32, shard_size=2000)
    # sh.save_shards()
