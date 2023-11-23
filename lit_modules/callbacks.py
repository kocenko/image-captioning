import torch
from torchvision.io import read_image
from lightning.pytorch.callbacks import Callback

from evaluation.extract_heads import aggregate_heads, extract_decoder_heads, extract_encoder_heads
from evaluation.visualizing import plot_self_attention, plot_cross_attention, plot_captioned_image
from evaluation.caption_generator import CaptionGenerator


class GenerateCaption(Callback):
    def __init__(self, sample_image_path: str, tokenizer, image_transform, vocab_size, device):
        super().__init__()
        self.si = sample_image_path
        self.tk = tokenizer
        self.it = image_transform
        self.vs = vocab_size
        self.dv = device

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        tensorboard = pl_module.logger.experiment

        image = read_image(self.si).permute(1, 2, 0)
        generator = CaptionGenerator(pl_module.model, self.tk, self.it, self.vs, self.dv)
        raw_caption = generator.generate_beam_search(self.si, 3)
        generated_caption = self.tk.decode(raw_caption)

        captioned_fig = plot_captioned_image(image, generated_caption)
        tensorboard.add_figure("captioned_image", captioned_fig)

        # Refitting the model
        dummy_caption = torch.tensor(raw_caption, device=self.dv).unsqueeze(0)
        dummy_image = self.it.transform(self.it.read_image(self.si).unsqueeze(0).to(self.dv))
        pl_module.model(dummy_image, dummy_caption)

        encoder_heads = extract_encoder_heads(pl_module.model)
        transformed_image = self.it.transform(image.permute(2, 0, 1))
        self_att_fig = plot_self_attention(transformed_image, encoder_heads, 3, 4, 14, 16, True)
        tensorboard.add_figure("self_attention", self_att_fig)

        decoder_heads = extract_decoder_heads(pl_module.model)
        aggregated_heads = aggregate_heads(decoder_heads, method="mean")
        cross_att_fig = plot_cross_attention(
            transformed_image, raw_caption, aggregated_heads, 8, 16, 14, self.tk.decode_map, True
        )
        tensorboard.add_figure("cross_attention", cross_att_fig)
