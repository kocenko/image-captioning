from typing import Any
import torch.cuda

from lightning.pytorch.callbacks import EarlyStopping

from lit_modules.lit_data import DataModule
from lit_modules.lit_model import ModelModule
from lit_modules.callbacks import GenerateCaption
from model.transformer import CaptionTransformer
from data_processing.image_transforms import ImageTransforms
from data_processing.feature_extractor import FeatureExtractor


def setup_model(
    model_config: dict, dataset_config: dict, sample_images: list[str]
) -> tuple[ModelModule, DataModule, dict, list[Any]]:
    dataset_name = dataset_config["name"]
    dataset_paths = dataset_config["dataset_paths"]

    hyperparameters = model_config["hyperparameters"]
    image_embedding_size = hyperparameters["encoder_sequence_size"]
    feature_extractor = model_config["feature_extractor"]
    slice_layer_name = model_config.get("layer_name", None)

    # Setting up image transformations
    image_transform = ImageTransforms(feature_extractor)
    extractor = FeatureExtractor(feature_extractor, image_transform)
    if feature_extractor != 'vit':
        extractor.slice_net(slice_layer_name, overwrite_model=True)
    if torch.cuda.is_available():
        extractor.model.cuda()

    # Setting up DataModule responsible for managing input data
    lit_data_module = DataModule(
        dataset_name,
        dataset_paths,
        image_transform,
        hyperparameters["max_caption_length"],
        hyperparameters["vocabulary_size"],
        hyperparameters["batches"],
    )

    ct = CaptionTransformer(lit_data_module.tokenizer, image_transform, extractor, **hyperparameters)
    lit_model = ModelModule(f"{feature_extractor} - {dataset_name}", ct, hyperparameters["learning_rate"])

    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    caption_gen = GenerateCaption(sample_images, image_embedding_size)
    callbacks = [early_stopping, caption_gen]

    return lit_model, lit_data_module, hyperparameters, callbacks
