import yaml

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

from lit_modules.lit_data import DataModule
from lit_modules.lit_model import ModelModule
from lit_modules.callbacks import GenerateCaption
from model.transformer import CaptionTransformer


def main():
    config_file = "configs/patching_small_flickr8k_surfing_homepc.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    dataset_name = config["dataset_name"]
    dataset_paths = config["dataset_paths"]
    hyperparameters = config["hyperparameters"]
    sample_image = config["sample_image"]
    checkpoints_folder = config["checkpoints_folder"]
    pretrained_weights_path = config.get("pretrained_weights_path", False)

    lit_data_module = DataModule(
        dataset_name,
        dataset_paths,
        hyperparameters["max_caption_length"] + 1,
        hyperparameters["vocabulary_size"],
        hyperparameters["image_size"],
        hyperparameters["batches"],
    )

    # Updating dependent hyperparameters
    hyperparameters["counter"] = lit_data_module.tokenizer.counter
    hyperparameters["encode_map"] = lit_data_module.tokenizer.encode_map

    ct = CaptionTransformer(**hyperparameters)
    if pretrained_weights_path:
        ct.load_weights(pretrained_weights_path)

    lit_model = ModelModule(ct, hyperparameters["encode_map"], hyperparameters["learning_rate"])
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    caption_gen = GenerateCaption(
        sample_image,
        lit_data_module.tokenizer,
        lit_data_module.transform,
        hyperparameters["vocabulary_size"],
    )

    trainer = Trainer(
        default_root_dir=checkpoints_folder,
        max_epochs=hyperparameters["epochs"],
        limit_train_batches=hyperparameters["steps_per_epoch"],
        limit_val_batches=hyperparameters["eval_iterations"],
        callbacks=[early_stopping, caption_gen],
        enable_model_summary=True,
    )
    trainer.fit(lit_model, lit_data_module)


if __name__ == "__main__":
    main()
