import yaml

from lightning import Trainer
from lightning.pytorch.callbacks import EarlyStopping

from lit_modules.lit_data import DataModule
from lit_modules.lit_model import ModelModule
from lit_modules.callbacks import GenerateCaption
from model.transformer import CaptionTransformer
from data_processing.image_transforms import ImageTransforms
from data_processing.feature_extractor import FeatureExtractor


def main():
    # Reading configuration data from yaml file
    config_file = "configs/mobilenet_small_flickr8k_surfing_homepc.yaml"
    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    dataset_name = config["dataset_name"]
    dataset_paths = config["dataset_paths"]
    hyperparameters = config["hyperparameters"]
    sample_image = config["sample_image"]
    checkpoints_folder = config["checkpoints_folder"]
    pretrained_weights_path = config.get("pretrained_weights_path", False)
    feature_extractor = config.get("feature_extractor", False)
    slice_layer_name = config.get("layer_name", None)
    hyperparameters["cross_att_key_dim"] = config.get("feature_maps_dim", hyperparameters["embeddings"])
    image_embedding_size = config.get("feature_maps_size", 14)  # For image_size: 224, patch_size: 16

    # Setting up image transformations
    extractor = None
    if feature_extractor:
        image_transform = ImageTransforms(hyperparameters["image_size"], feature_extractor)
        extractor = FeatureExtractor(feature_extractor, image_transform)
        extractor.slice_net(slice_layer_name, overwrite_model=True)
    else:
        image_transform = ImageTransforms(hyperparameters["image_size"])

    # Setting up DataModule responsible for managing input data
    lit_data_module = DataModule(
        dataset_name,
        dataset_paths,
        image_transform,
        hyperparameters["max_caption_length"],
        hyperparameters["vocabulary_size"],
        hyperparameters["batches"],
    )

    # Updating dependent hyperparameters
    hyperparameters["counter"] = lit_data_module.tokenizer.counter
    hyperparameters["encode_map"] = lit_data_module.tokenizer.encode_map

    if not extractor:
        ct = CaptionTransformer(lit_data_module.tokenizer, image_transform, **hyperparameters)
    else:
        ct = CaptionTransformer(lit_data_module.tokenizer, image_transform, extractor, **hyperparameters)

    if pretrained_weights_path and not extractor:
        ct.load_weights(pretrained_weights_path)

    lit_model = ModelModule(ct, hyperparameters["encode_map"], hyperparameters["learning_rate"])

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    caption_gen = GenerateCaption(sample_image, image_embedding_size, not feature_extractor)

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
