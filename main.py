import os

from data_processing.tokenizer import Tokenizer
from data_processing.custom_dataset import ImageCaptionDataset
from data_processing.dataset_reader import load_flickr8k
from data_processing.image_transforms import ImageTransforms
from model.transformer import CaptionTransformer
from model.train import Trainer

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np


def main():
    sample_image = "./evaluation/sample_images/surfing.jpg"
    checkpoints_folder = "./evaluation/checkpoints"
    summary_folder = "./evaluation/summary"
    pretrained_weights_path = "../pretrained_weights/pytorch_model.bin"

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
        "max_caption_length": 30,
        "vocabulary_size": 5000,
        "banned_tokens": ["<unknown>", "<start>", ""],
        "embeddings": 768,
        "dropout_rate": 0.6,
        "patch_size": 16,
        "image_size": (224, 224),
        "shift_pixels": 5,
        "encoder_layers": 12,
        "decoder_layers": 4,
        "learning_rate": 1e-4,
        "epochs": 100,
        "heads_num": 12,
        "eval_iterations": 20,
        "eval_per_epoch": 10,
        "device": "cpu",
    }

    if torch.cuda.is_available():
        hyperparameters["device"] = "cuda"
        print("Will be using CUDA!!!")
    device = hyperparameters["device"]

    tk = Tokenizer(
        [caption for _, caption in train_ds],
        max_sequence_size=hyperparameters["max_caption_length"] + 1,
        vocabulary_size=hyperparameters["vocabulary_size"],
    )

    # Updating dependent hyperparameters
    hyperparameters["counter"] = tk.counter
    hyperparameters["encode_map"] = tk.encode_map

    # Preparing folders for logging
    for path in [checkpoints_folder, summary_folder]:
        if not os.path.exists(path):
            os.makedirs(path)

    image_size = hyperparameters["image_size"]
    it = ImageTransforms(image_size)
    datasets = [
        ImageCaptionDataset(train_ds, tk, it, device),
        ImageCaptionDataset(valid_ds, tk, it, device),
        ImageCaptionDataset(test_ds, tk, it, device),
    ]
    ct = CaptionTransformer(**hyperparameters)
    # ct.load_weights(pretrained_weights_path)

    torch.manual_seed(2013)
    for name, val in ct.named_parameters():
        val.data.copy_(torch.rand_like(val))

    caption = train_ds[0][1]
    img_custom = it.transform(it.read_image(train_ds[0][0]).unsqueeze(0))
    with torch.no_grad():
        ct(img_custom, torch.tensor(tk.encode(caption)[:-1]).unsqueeze(0))

    # wr = SummaryWriter(summary_folder)
    # trainer = Trainer(
    #     ct,
    #     tk,
    #     it,
    #     hyperparameters["vocabulary_size"],
    #     datasets,
    #     test_ds,
    #     checkpoints_folder,
    #     sample_image,
    #     wr,
    #     hyperparameters,
    # )
    # trainer.train()


if __name__ == "__main__":
    main()
