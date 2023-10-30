import os

from image_captioning.transformer import Decoder
from image_captioning.tokenizer import Tokenizer
from image_captioning.feature_extractor import FeatureExtractor
from image_captioning.dataset import Sharder, load_flickr8k, custom_dataloader
from image_captioning.train import Trainer

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter


def calc_single_loss(tk, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Calculates a loss of a single predictions-labels pair

    Args:
        predictions (Tensor): captions as an output from the decoder (as logits)
        labels (Tensor): ground truth captions

    Returns:
        Tensor as an output of the cross entropy with logarithmic softmax. Calculated for the whole batch.
    """

    predictions = predictions.transpose(-2, -1)
    loss = F.cross_entropy(predictions, labels, reduction="none")

    mask = (labels != tk.encode_map[Tokenizer.empty_token]) & (loss < 1e8)
    mask = mask.float()

    loss = loss * mask
    loss = torch.sum(loss) / torch.sum(mask)
    return loss


def main():
    tokens_path = "./dataset/Flickr8k.token.txt"
    train_path = "./dataset/Flickr_8k.trainImages.txt"
    valid_path = "./dataset/Flickr_8k.devImages.txt"
    test_path = "./dataset/Flickr_8k.testImages.txt"
    images_path = "./dataset/images"
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
        "epochs": 50,
        "blocks_number": 2,
        "heads_number": 2,
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

    tk = Tokenizer([caption for _, caption in train_ds])
    sh = Sharder(tk, fe, batch_size=hyperparameters["batches"], shard_size=2000, device=hyperparameters["device"])
    # sh.save_shards(train_ds, "train", "shards/train")
    # sh.save_shards(valid_ds, "valid", "shards/valid")
    # sh.save_shards(test_ds, "test", "shards/test")
    sh.load_shards(["shards/train", "shards/valid", "shards/test"], ["train", "valid", "test"])

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

    dc = Decoder(**hyperparameters)
    loader = custom_dataloader('train', sh, hyperparameters["batches"])
    optimizer = torch.optim.AdamW(dc.parameters(), lr=hyperparameters["learning_rate"])

    device = hyperparameters["device"]
    for (x1, x2, y) in loader:
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = dc(x1, x2)
        loss = calc_single_loss(tk, logits, y)
        print(loss.item())
        loss.backward()
        optimizer.step()

    # wr = SummaryWriter(summary_folder)
    # trainer = Trainer(tk, fe, sh, checkpoints_folder, sample_image, wr, hyperparameters)
    # trainer.train()


if __name__ == "__main__":
    main()
