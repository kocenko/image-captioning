import os

from data_processing.tokenizer import Tokenizer
from data_processing.dataset import ImageCaptionDataset
from data_processing.loader import load_flickr8k
from model.transformer import CaptionTransformer
from model.train import Trainer

import torch
from torch.utils.tensorboard import SummaryWriter


def main():

    sample_image = "./evaluation/sample_images/surfing.jpg"
    checkpoints_folder = "./evaluation/checkpoints"
    summary_folder = "./evaluation/summary"

    # Option 1.
    tokens_path = "../dataset/Flickr8k.token.txt"
    train_path = "../dataset/Flickr_8k.trainImages.txt"
    valid_path = "../dataset/Flickr_8k.devImages.txt"
    test_path = "../dataset/Flickr_8k.testImages.txt"
    images_path = "../dataset/images"
    train_ds, valid_ds, test_ds = load_flickr8k(tokens_path, train_path, valid_path, test_path, images_path)

    # # Option 2.
    # tokens_path = "../dataset/flickr30k/captions.txt"
    # images_path = "../dataset/flickr30k/Images"
    # train_ds, valid_ds, test_ds = load_flickr30k(tokens_path, images_path)

    hyperparameters = {
        "batches": 32,
        "max_caption_length": 60,
        "vocabulary_size": 5000,
        "banned_tokens": ["<unknown>", "<start>", ""],
        "embeddings": 256,
        "dropout_rate": 0.5,
        "patch_size": 16,
        "image_size": (224, 224),
        "shift_pixels": (5, 5),
        "encoder_layers": 4,
        "decoder_layers": 4,
        "learning_rate": 1e-4,
        "epochs": 100,
        "heads_num": 2,
        "eval_iterations": 20,
        "eval_per_epoch": 10,
        "device": "cpu"
    }

    if torch.cuda.is_available():
        hyperparameters["device"] = "cuda"
        print("Will be using CUDA!!!")
    device = hyperparameters["device"]

    tk = Tokenizer(
        [caption for _, caption in train_ds],
        max_sequence_size=hyperparameters["max_caption_length"]+1,
        vocabulary_size=hyperparameters["vocabulary_size"]
    )

    # Updating dependent hyperparameters
    hyperparameters["counter"] = tk.counter
    hyperparameters["encode_map"] = tk.encode_map

    # Preparing folders for logging
    for path in [checkpoints_folder, summary_folder]:
        if not os.path.exists(path):
            os.makedirs(path)

    image_size = hyperparameters["image_size"]
    datasets = [
        ImageCaptionDataset(train_ds, image_size, tk, device),
        ImageCaptionDataset(valid_ds, image_size, tk, device),
        ImageCaptionDataset(test_ds, image_size, tk, device)
    ]
    ct = CaptionTransformer(**hyperparameters)
    wr = SummaryWriter(summary_folder)
    trainer = Trainer(ct, tk, datasets, checkpoints_folder, sample_image, wr, hyperparameters)
    trainer.train()


if __name__ == "__main__":
    main()
