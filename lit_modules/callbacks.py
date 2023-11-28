import torch
from lightning.pytorch.callbacks import Callback

import data_processing.image_transforms
from evaluation.extract_heads import aggregate_heads, extract_decoder_heads, extract_encoder_heads
from evaluation.visualizing import (
    plot_self_attention,
    plot_cross_attention,
    plot_captioned_image,
    visualize_candidates_graph,
)
from data_processing.tokenizer import Tokenizer
from data_processing.image_transforms import ImageTransforms


class GenerateCaption(Callback):
    def __init__(
        self,
        image_path: str,
        image_embedding_size: int,
        show_self_attention: bool,
    ):
        super().__init__()
        self.image_path = image_path
        self.image_embedding_size = image_embedding_size
        self.show_self_attention = show_self_attention

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        tensorboard = pl_module.logger.experiment

        # Visualizing root image with the generated caption
        raw_caption, beam_history = pl_module.model.generate_beam_search(self.image_path, 3)
        generated_caption = pl_module.model.tokenizer.decode(raw_caption)
        print(generated_caption)
        # image = pl_module.model.image_transform.read_image(self.image_path).permute(1, 2, 0)
        # captioned_fig = plot_captioned_image(image, generated_caption)
        # tensorboard.add_figure("captioned_image", captioned_fig, trainer.current_epoch)
        # print(generated_caption)

        visualize_candidates_graph(beam_history, pl_module.model.tokenizer.decode)

        # Refitting the model
        # dummy_caption = torch.tensor(raw_caption).unsqueeze(0)
        # dummy_image = self.image_transform.transform(self.image_transform.read_image(self.image_path).unsqueeze(0))
        # pl_module.model(dummy_image, dummy_caption)
        #
        # dummy_image = self.image_transform.denormalize(dummy_image.squeeze(0))
        # if self.show_self_attention:
        #     encoder_heads = extract_encoder_heads(pl_module.model)
        #     self_att_fig = plot_self_attention(dummy_image, encoder_heads, 3, 4, self.image_embedding_size)
        #     tensorboard.add_figure("self_attention", self_att_fig)

        # decoder_heads = extract_decoder_heads(pl_module.model)
        # aggregated_heads = aggregate_heads(decoder_heads, method="sum")
        # cross_att_fig = plot_cross_attention(
        #     dummy_image, raw_caption, aggregated_heads, 8, self.image_embedding_size, self.tokenizer.decode_map
        # )
        # tensorboard.add_figure("cross_attention", cross_att_fig)
