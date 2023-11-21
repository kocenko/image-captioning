import matplotlib.pyplot as plt
from torchvision.io import read_image
from lightning.pytorch.callbacks import Callback

from evaluation.extract_heads import extract_encoder_heads
from evaluation.visualizing import plot_self_attention
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
        generator = CaptionGenerator(
            pl_module.model,
            self.tk,
            self.it,
            self.vs,
            self.dv
        )
        generated_caption = generator.generate_beam_search(self.si, 3)
        image = read_image(self.si).permute(1, 2, 0)
        fig, ax = plt.subplots(1)
        plt.axis('off')
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
        tensorboard.add_figure('captioned_image', fig)

        # encoder_heads = extract_encoder_heads(pl_module.model)
        # plot_self_attention(self.it.transform(image.permute(2, 0, 1)), encoder_heads, 3, 4, 16)

