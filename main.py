import argparse
import yaml
from lightning import Trainer
from setup_model import setup_model


def main(model: str, dataset: str):
    with open("configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    lit_model, lit_data_module, hyperparameters, callbacks = setup_model(
        config["model"][model], config["dataset"][dataset], config["sample_images"]
    )
    trainer = Trainer(
        default_root_dir=config["checkpoints_folder"],
        max_epochs=hyperparameters["epochs"],
        limit_train_batches=hyperparameters["steps_per_epoch"],
        limit_val_batches=hyperparameters["eval_iterations"],
        callbacks=callbacks,
        enable_model_summary=True,
    )
    trainer.fit(lit_model, lit_data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="ImageCaptioning")
    parser.add_argument("-m", "--model", required=True, choices=["vit", "vgg", "mobilenet"])
    parser.add_argument("-d", "--dataset", required=True, choices=["vizwiz", "flickr8k", "flickr30k"])
    args = parser.parse_args()
    main(args.model, args.dataset)
