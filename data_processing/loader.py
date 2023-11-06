import os
import random
import tqdm


def load_flickr8k(
    tokens_path: str, train_path: str, valid_path: str, test_path: str, images_path: str
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    print("Loading Flickr8k dataset...")
    
    # Get the list of all images' paths
    images_paths = os.listdir(images_path)

    # Read file paths to every group and filter out non-existing paths
    train_paths, valid_paths, test_paths = [], [], []
    for path_to_file, array in zip([train_path, valid_path, test_path], [train_paths, valid_paths, test_paths]):
        with open(path_to_file, "r") as f:
            content = f.read()
        array.extend([path for path in tqdm.tqdm(content.splitlines()) if path in images_paths])

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


def load_flickr30k(
    tokens_path: str, images_path: str, split_ratio: tuple[float, float, float] = (0.7, 0.2, 0.1), seed: int = 42
) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    print("Loading Flickr30k dataset...")

    # Get the list of all images' paths
    images_paths = os.listdir(images_path)

    # Filtering captions if the images do not exist
    with open(tokens_path, "r") as f:
        content = f.read()

    whole_dataset = []
    for line in tqdm.tqdm(content.splitlines()):
        split_list = line.split(',', maxsplit=1)
        if len(split_list) == 2 and split_list[0] in images_paths:
            whole_dataset.append((os.path.join(images_path, split_list[0]), split_list[1]))

    # Shuffling
    random.seed(seed)
    random.shuffle(whole_dataset)

    # Split the shuffled data into three groups
    group_sizes = [int(ratio * len(whole_dataset)) for ratio in split_ratio]
    groups = []
    start = 0
    for size in group_sizes:
        group = whole_dataset[start : start + size]
        groups.append(group)
        start += size

    return groups[0], groups[1], groups[2]
