import matplotlib.pyplot as plt
from torchvision.io import read_image
from lightning.pytorch.callbacks import Callback

from evaluation.extract_heads import extract_encoder_heads, extract_decoder_heads, aggregate_heads
from evaluation.visualizing import plot_self_attention, plot_cross_attention
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
        generator = CaptionGenerator(pl_module.model, self.tk, self.it, self.vs, self.dv)
        raw_caption = generator.generate_beam_search(self.si, 3)
        generated_caption = self.tk.decode(raw_caption)
        image = read_image(self.si).permute(1, 2, 0)
        fig, ax = plt.subplots(1)
        plt.axis("off")
        ax.imshow(image)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=1.0)
        ax.text(
            image.shape[1] // 2,
            image.shape[0] + 0.02,
            f"{generated_caption[7:-5]}",
            ha="center",
            va="center",
            size=10,
            bbox=bbox_props,
        )
        tensorboard = pl_module.logger.experiment
        tensorboard.add_figure("captioned_image", fig)

        transformed_image = self.it.transform(image.permute(2, 0, 1))

        encoder_heads = extract_encoder_heads(pl_module.model)
        self_att_fig = plot_self_attention(transformed_image, encoder_heads, 3, 4, 14, patch_size=16)
        tensorboard.add_figure("self_attention", self_att_fig)

        decoder_heads = extract_decoder_heads(pl_module.model)[-1]  # Last cross attention layer
        aggregated_heads = aggregate_heads(decoder_heads)
        cross_att_fig = plot_cross_attention(
            transformed_image, raw_caption, aggregated_heads, 8, 16, 14, self.tk.decode_map
        )
        tensorboard.add_figure("cross_attention", cross_att_fig)
