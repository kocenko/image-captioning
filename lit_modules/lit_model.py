import torch
from torchtext.data.metrics import bleu_score
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from lightning import LightningModule

from data_processing.tokenizer import Tokenizer
from model.transformer import CaptionTransformer


class ModelModule(LightningModule):
    def __init__(self, config_file: str, model: CaptionTransformer, learning_rate: float):
        super().__init__()
        self.config_file = config_file
        self.model = model
        self.encode_map = self.model.tokenizer.encode_map
        self.lr = learning_rate
        self.save_hyperparameters(ignore=["model"])

    def forward(self, image: torch.Tensor, caption: torch.Tensor) -> torch.Tensor:
        logits = self.model(image, caption)
        return logits

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def loss_function(self, logits: torch.Tensor, target: torch.Tensor):
        padding_token = self.encode_map[Tokenizer.empty_token]
        loss = cross_entropy(logits, target, ignore_index=padding_token, reduction="none")
        mask = (target != padding_token) & (loss < 1e8)
        loss = loss * mask
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

    def masked_accuracy(self, logits: torch.Tensor, target: torch.Tensor):
        padding_token = self.encode_map[Tokenizer.empty_token]
        logits = logits.transpose(-2, -1)
        mask = target != padding_token
        predictions = torch.argmax(logits, dim=-1)
        match = predictions == target
        acc = match * mask
        acc = torch.sum(acc) / torch.sum(mask)
        return acc

    def training_step(self, batch, batch_idx):
        image, caption_sample, caption_target = batch
        logits = self(image, caption_sample)

        loss = self.loss_function(logits, caption_target)
        self.log("train_loss", loss)

        acc = self.masked_accuracy(logits, caption_target)
        self.log("train_accuracy", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        image, caption_sample, caption_target = batch

        # Calculating bleu score
        # reference = [self.model.tokenizer.decode(caption.tolist()).split()[:-1] for caption in caption_target]
        # hypothesis = [
        #     self.model.tokenizer.decode(self.model.generate_beam_search(img.unsqueeze(0), 3)[0]).split()[1:-1]
        #     for img in image
        # ]
        # for bleu_n in [1, 2, 3, 4]:
        #     score = bleu_score(
        #         candidate_corpus=hypothesis, references_corpus=reference, max_n=bleu_n, weights=[0.25] * bleu_n
        #     )
        #     self.log(f"bleu-{bleu_n}", score)

        logits = self(image, caption_sample)

        loss = self.loss_function(logits, caption_target)
        self.log("val_loss", loss)

        acc = self.masked_accuracy(logits, caption_target)
        self.log("val_accuracy", acc)

        return loss
