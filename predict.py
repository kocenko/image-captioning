import torch
from setup_model import setup_model
from lit_modules.lit_model import ModelModule


def predict():
    checkpoints_path = 'checkpoints/lightning_logs/version_5/checkpoints/epoch=0-step=100.ckpt'
    config_file_path = "configs/mobilenet_small_flickr8k_bikes_homepc.yaml"
    sample_image = 'evaluation/sample_images/surfing.jpg'
    lit_model, lit_data_module, hyperparameters, checkpoints_folder, callbacks = setup_model(config_file_path)

    checkpoint = torch.load(checkpoints_path, map_location=lambda storage, loc: storage)
    params = checkpoint['hyper_parameters']
    if (config_file := params.get('config_file', None)) is not None and config_file != config_file_path:
        raise Exception(f'Expected {config_file}')

    model = ModelModule.load_from_checkpoint(
        checkpoints_path,
        config_file=config_file,
        model=lit_model.model,
        token_encode_map=lit_model.model.tokenizer.encode_map,
        learning_rate=params['learning_rate']
    )

    model.eval()
    caption = model.model.tokenizer.decode(model.model.generate_beam_search(sample_image, 3)[0])
    print(caption)


if __name__ == "__main__":
    predict()
