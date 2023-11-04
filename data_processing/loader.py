import os


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