import os

from image_captioning.tokenizer import Tokenizer
from image_captioning.feature_extractor import FeatureExtractor
from image_captioning.dataset import Sharder, load_flickr8k
from image_captioning.train import Trainer

import torch
from torch.utils.tensorboard import SummaryWriter


def main():
    tokens_path = "../dataset/Flickr8k.token.txt"
    train_path = "../dataset/Flickr_8k.trainImages.txt"
    valid_path = "../dataset/Flickr_8k.devImages.txt"
    test_path = "../dataset/Flickr_8k.testImages.txt"
    images_path = "../dataset/images"
    sample_image = "./sample_images/surfing.jpg"
    checkpoints_folder = "./checkpoints"
    summary_folder = "./summary"

    train_ds, valid_ds, test_ds = load_flickr8k(tokens_path, train_path, valid_path, test_path, images_path)

    hyperparameters = {
        "batches": 32,
        "split_lengths": (.7, .2, .1),
        "banned_tokens": ["<unknown>", "<start>", ""],
        "embeddings_number": 256,
        "dropout_rate": 0.5,
        "learning_rate": 1e-4,
        "epochs": 20,
        "blocks_number": 2,
        "heads_number": 2,
        "net_slice_index": "features.12",
        "eval_iterations": 10,
        "eval_per_epoch": 10,
        "device": "cpu"
    }

    if torch.cuda.is_available():
        hyperparameters["device"] = "cuda"
        print("Will be using CUDA!!!")

    fe = FeatureExtractor(model_name="mobilenet", device=hyperparameters["device"])
    if hyperparameters["net_slice_index"]:
        fe.slice_net(hyperparameters["net_slice_index"], overwrite_model=True)

    tk = Tokenizer([caption for _, caption in train_ds])

    batches = hyperparameters["batches"]
    sh = Sharder(tk, fe, batch_size=batches, shard_size=2000, device=hyperparameters["device"])
    wr = SummaryWriter(summary_folder)

    # Updating dependent hyperparameters
    hyperparameters["vocabulary_size"] = len(tk.word_list)
    hyperparameters["context_length"] = tk.max_length
    hyperparameters["image_channels"] = fe.feed(fe.get_image_from_file(sample_image).unsqueeze(0)).shape[1]
    hyperparameters["word_count"] = tk.counter
    hyperparameters["encode_map"] = tk.encode_map

    # Preparing folders for logging
    for path in [checkpoints_folder, summary_folder]:
        if not os.path.exists(path):
            os.makedirs(path)

    trainer = Trainer(tk, fe, sh, checkpoints_folder, sample_image, wr, hyperparameters)
    trainer.train()


if __name__ == "__main__":
    main()
