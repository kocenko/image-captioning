from lightning import Trainer
from setup_model import setup_model


def main():
    # Reading configuration data from yaml file
    config_file = "configs/mobilenet_small_flickr8k_bikes_homepc.yaml"
    lit_model, lit_data_module, hyperparameters, checkpoints_folder, callbacks = setup_model(config_file)
    trainer = Trainer(
        default_root_dir=checkpoints_folder,
        max_epochs=hyperparameters["epochs"],
        limit_train_batches=hyperparameters["steps_per_epoch"],
        limit_val_batches=hyperparameters["eval_iterations"],
        callbacks=callbacks,
        enable_model_summary=True,
    )
    trainer.fit(lit_model, lit_data_module)


if __name__ == "__main__":
    main()
