import os

from tokenizer import Tokenizer
from feature_extractor import FeatureExtractor
from dataset import Sharder
from train import Trainer

from torch.utils.tensorboard import SummaryWriter


def main():
    paths = {
        "dataset_folder": "dataset",
        "sample_images_folder": "imgs",
        "checkpoints_folder": "checkpoints",
        "summary_folder": "summary"
    }
    paths["images"] = os.path.join(paths["dataset_folder"], "images")
    paths["captions"] = os.path.join(paths["dataset_folder"], "captions.txt")
    paths["sample_image"] = os.path.join(paths["sample_images_folder"], "rooster.jpg")

    folders = [paths["dataset_folder"], paths["images"], paths["summary_folder"], paths["checkpoints_folder"]]
    for path in folders:
        if not os.path.exists(path):
            os.makedirs(path)

    with open(paths["captions"], "r") as f:
        raw_file = f.read()

    # Head size should be equal to embeddings_number // heads_number
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
        "head_size": 128,
        "net_slice_index": "features.12",
        "eval_iterations": 10,
        "eval_per_epoch": 10,
        "device": "cpu"
    }

    fe = FeatureExtractor(model_name="mobilenet", device=hyperparameters["device"])
    if hyperparameters["net_slice_index"]:
        fe.slice_net(hyperparameters["net_slice_index"], overwrite_model=True)
    tk = Tokenizer(raw_file, paths["images"], reduce=True)
    batches = hyperparameters["batches"]
    sh = Sharder(tk, fe, batch_size=batches, shard_size=2000, device=hyperparameters["device"])
    # sh.save_shards()
    wr = SummaryWriter(paths["summary_folder"])

    # Updating dependent hyperparameters
    hyperparameters["vocabulary_size"] = len(tk.word_list)
    hyperparameters["context_length"] = tk.max_length
    hyperparameters["image_channels"] = fe.feed(fe.get_image_from_file(paths["sample_image"]).unsqueeze(0)).shape[1]
    hyperparameters["word_count"] = tk.counter
    hyperparameters["encode_map"] = tk.encode_map

    trainer = Trainer(tk, fe, sh, paths["checkpoints_folder"], paths["sample_image"], wr, hyperparameters)
    trainer.train()


if __name__ == "__main__":
    main()
