import argparse
import yaml
import torch
from setup_model import setup_model
from lit_modules.lit_model import ModelModule


def predict(model: str, dataset: str):
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    dataset_name = config["dataset"]["name"]
    feature_extractor = config["model"].get("feature_extractor", "patching")
    checkpoints_path = "../checkpoints/lightning_logs/version_0/checkpoints/epoch=96-step=9700.ckpt"
    sample_image = "../ducky.jpg"
    lit_model, lit_data_module, hyperparameters, callbacks = setup_model(
        config["model"][model], config["dataset"][dataset], config["sample_images"]
    )

    checkpoint = torch.load(checkpoints_path, map_location=lambda storage, loc: storage)
    params = checkpoint["hyper_parameters"]
    if (
        config_file := params.get("config_name", None)
    ) is not None and config_file != f"{feature_extractor} - {dataset_name}":
        raise Exception(f"Expected {config_file}")

    model = ModelModule.load_from_checkpoint(
        checkpoints_path, config_file=config_file, model=lit_model.model, learning_rate=params["learning_rate"]
    )

    model.eval()
    caption = model.model.tokenizer.decode(model.model.generate_beam_search(sample_image, 3)[0])
    print(caption)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ImageCaptioning-Predict")
    parser.add_argument("-m", "--model", required=True, choices=["patching", "vgg", "mobilenet"])
    parser.add_argument("-d", "--dataset", required=True, choices=["vizwiz", "flickr8k", "flickr30k"])
    args = parser.parse_args()
    predict(args.model, args.dataset)
