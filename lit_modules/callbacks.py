import matplotlib.pyplot as plt
from torchvision.io import read_image
from lightning.pytorch.callbacks import Callback

from evaluation.caption_generator import CaptionGenerator


class GenerateCaption(Callback):
    def __init__(self, sample_image_path: str, tokenizer, image_transform, vocab_size, device):
        super().__init__()
        self.si = sample_image_path
        self.tk = tokenizer
        self.it = image_transform
        self.vs = vocab_size
        self.dv = device

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        generator = CaptionGenerator(
            pl_module.model,
            self.tk,
            self.it,
            self.vs,
            self.dv
        )
        generated_caption = generator.generate_beam_search(self.si, 3)
        print(generated_caption)
