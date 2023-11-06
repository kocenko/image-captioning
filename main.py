import os

from model.transformer import Decoder
from data_processing.tokenizer import Tokenizer
from data_processing.feature_extractor import FeatureExtractor
from data_processing.dataset import DataCachingManager, ImageCaptionDataset
from data_processing.loader import load_flickr8k, load_flickr30k
from model.train import Trainer

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main():

    sample_image = "./evaluation/sample_images/surfing.jpg"
    checkpoints_folder = "./evaluation/checkpoints"
    summary_folder = "./evaluation/summary"

    # # Option 1.
    # tokens_path = "../dataset/Flickr8k.token.txt"
    # train_path = "../dataset/Flickr_8k.trainImages.txt"
    # valid_path = "../dataset/Flickr_8k.devImages.txt"
    # test_path = "../dataset/Flickr_8k.testImages.txt"
    # images_path = "../dataset/images"
    # train_ds, valid_ds, test_ds = load_flickr8k(tokens_path, train_path, valid_path, test_path, images_path)

    # Option 2.
    tokens_path = "../dataset/flickr30k/captions.txt"
    images_path = "../dataset/flickr30k/Images"
    train_ds, valid_ds, test_ds = load_flickr30k(tokens_path, images_path)


    hyperparameters = {
        "batches": 32,
        "context_length": 60,
        "vocabulary_size": 5000,
        "banned_tokens": ["<unknown>", "<start>", ""],
        "embeddings_number": 256,
        "dropout_rate": 0.5,
        "learning_rate": 1e-4,
        "epochs": 100,
        "blocks_number": 2,
        "heads_number": 2,
        "head_size": 128,
        "net_slice_index": "features.12",
        "eval_iterations": 20,
        "eval_per_epoch": 10,
        "device": "cpu"
    }

    if torch.cuda.is_available():
        hyperparameters["device"] = "cuda"
        print("Will be using CUDA!!!")

    fe = FeatureExtractor(model_name="mobilenet", device=hyperparameters["device"])
    if hyperparameters["net_slice_index"]:
        fe.slice_net(hyperparameters["net_slice_index"], overwrite_model=True)

    tk = Tokenizer([caption for _, caption in train_ds], max_sequence_size=hyperparameters["context_length"], vocabulary_size=hyperparameters["vocabulary_size"])

    # Option 1.
    sh = DataCachingManager(tk, fe, batch_size=hyperparameters["batches"], shard_size=2000, device=hyperparameters["device"])
    # sh.save_shards(train_ds, "train", "shards/train")
    # sh.save_shards(valid_ds, "valid", "shards/valid")
    # sh.save_shards(test_ds, "test", "shards/test")
    sh.load_shards(["shards/train", "shards/valid", "shards/test"], ["train", "valid", "test"])

    # # Option 2.
    # train_set = ImageCaptionDataset(train_ds, tk, fe, hyperparameters["device"])
    # valid_set = ImageCaptionDataset(valid_ds, tk, fe, hyperparameters["device"])
    # sh = (train_set, valid_set)

    # Updating dependent hyperparameters
    hyperparameters["image_channels"] = fe.feed(fe.get_image_from_file(sample_image).unsqueeze(0)).shape[1]
    hyperparameters["word_count"] = tk.counter
    hyperparameters["encode_map"] = tk.encode_map

    # Preparing folders for logging
    for path in [checkpoints_folder, summary_folder]:
        if not os.path.exists(path):
            os.makedirs(path)

    # mock_image = torch.ones((32, 576, 7, 7)).to(torch.float32).to("cuda")
    # mock_caption = torch.ones((32, 20)).to(torch.int32).to("cuda")
    # dc = Decoder(**hyperparameters)

    # for param in dc.parameters():
    #     torch.nn.init.constant_(param, 2.0)

    # for name, param in dc.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name} --- {param.data.shape}")
    #         print(param.data.detach().cpu().numpy())
    #         print()
    #         print()

    # dc.eval()
    # dc(mock_image, mock_caption)

    wr = SummaryWriter(summary_folder)
    trainer = Trainer(tk, fe, sh, checkpoints_folder, sample_image, wr, hyperparameters)
    trainer.train()


if __name__ == "__main__":
    main()
