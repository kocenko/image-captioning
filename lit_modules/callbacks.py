import torch
from lightning.pytorch.callbacks import Callback

import data_processing.image_transforms
from evaluation.extract_heads import aggregate_heads, extract_decoder_heads, extract_encoder_heads
from evaluation.visualizing import plot_self_attention, plot_cross_attention, plot_captioned_image, visualize_candidates_graph
from evaluation.caption_generator import CaptionGenerator
from data_processing.tokenizer import Tokenizer
from data_processing.image_transforms import ImageTransforms


class GenerateCaption(Callback):
    def __init__(
        self,
        image_path: str,
        tokenizer: Tokenizer,
        image_transform: ImageTransforms,
        vocab_size: int,
        image_embedding_size: int,
        show_self_attention: bool,
    ):
        super().__init__()
        self.image_path = image_path
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.vocabulary_size = vocab_size
        self.image_embedding_size = image_embedding_size
        self.show_self_attention = show_self_attention

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        tensorboard = pl_module.logger.experiment

        image = self.image_transform.read_image(self.image_path).permute(1, 2, 0)
        generator = CaptionGenerator(pl_module.model, self.tokenizer, self.image_transform, self.vocabulary_size)
        raw_caption = generator.generate(self.image_path, temperature=0.0)

        # visualize_candidates_graph(root_node, self.tokenizer.decode)

        generated_caption = self.tokenizer.decode(raw_caption)
        print(generated_caption)
        captioned_fig = plot_captioned_image(image, generated_caption)
        tensorboard.add_figure("captioned_image", captioned_fig)

        # Refitting the model
        dummy_caption = torch.tensor(raw_caption).unsqueeze(0)
        dummy_image = self.image_transform.transform(self.image_transform.read_image(self.image_path).unsqueeze(0))
        pl_module.model(dummy_image, dummy_caption)

        dummy_image = self.image_transform.denormalize(dummy_image.squeeze(0))
        if self.show_self_attention:
            encoder_heads = extract_encoder_heads(pl_module.model)
            self_att_fig = plot_self_attention(dummy_image, encoder_heads, 3, 4, self.image_embedding_size)
            tensorboard.add_figure("self_attention", self_att_fig)

        decoder_heads = extract_decoder_heads(pl_module.model)
        aggregated_heads = aggregate_heads(decoder_heads, method="sum")
        cross_att_fig = plot_cross_attention(
            dummy_image, raw_caption, aggregated_heads, 8, self.image_embedding_size, self.tokenizer.decode_map
        )
        tensorboard.add_figure("cross_attention", cross_att_fig)
