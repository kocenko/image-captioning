import torch

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

from lit_modules.lit_data import DataModule
from lit_modules.lit_model import ModelModule
from lit_modules.callbacks import GenerateCaption
from model.transformer import CaptionTransformer


def main():
    sample_image = "./evaluation/sample_images/surfing.jpg"
    pretrained_weights_path = "../pretrained_weights/pytorch_model.bin"
    # checkpoints_folder = "./evaluation/checkpoints"
    # summary_folder = "./evaluation/summary"

    dataset_name = "flickr8k"
    dataset_paths = {}
    if dataset_name == "flickr8k":
        dataset_paths = {
            "tokens_path": "../dataset/Flickr8k.token.txt",
            "train_path": "../dataset/Flickr_8k.trainImages.txt",
            "valid_path": "../dataset/Flickr_8k.devImages.txt",
            "test_path": "../dataset/Flickr_8k.testImages.txt",
            "images_path": "../dataset/images",
        }
    elif dataset_name == "flickr30k":
        dataset_paths = {
            "tokens_path": "../dataset/flickr30k/captions.txt",
            "images_path": "../dataset/flickr30k/Images",
        }

    hyperparameters = {
        "batches": 32,
        "max_caption_length": 40,
        "vocabulary_size": 5000,
        "banned_tokens": ["<unknown>", "<start>", ""],
        "embeddings": 768,
        "dropout_rate": 0.3,
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

    lit_data_module = DataModule(
        dataset_name,
        dataset_paths,
        hyperparameters["max_caption_length"] + 1,
        hyperparameters["vocabulary_size"],
        hyperparameters["image_size"],
        hyperparameters["batches"],
        hyperparameters["device"],
    )

    # Updating dependent hyperparameters
    hyperparameters["counter"] = lit_data_module.tokenizer.counter
    hyperparameters["encode_map"] = lit_data_module.tokenizer.encode_map

    # # Preparing folders for logging
    # for path in [checkpoints_folder, summary_folder]:
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    ct = CaptionTransformer(**hyperparameters)
    ct.load_weights(pretrained_weights_path)

    lit_model = ModelModule(ct, hyperparameters["encode_map"], hyperparameters["learning_rate"])

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    caption_gen = GenerateCaption(
        sample_image,
        lit_data_module.tokenizer,
        lit_data_module.transform,
        hyperparameters["vocabulary_size"],
        device=hyperparameters["device"],
    )
    trainer = Trainer(
        max_epochs=hyperparameters["epochs"],
        val_check_interval=1 / hyperparameters["eval_per_epoch"],
        limit_val_batches=hyperparameters["eval_iterations"],
        callbacks=[early_stopping, caption_gen],
    )
    trainer.fit(lit_model, lit_data_module)


if __name__ == "__main__":
    main()
