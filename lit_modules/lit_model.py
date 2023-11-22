import torch
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from lightning import LightningModule

from data_processing.tokenizer import Tokenizer
from model.transformer import CaptionTransformer


class ModelModule(LightningModule):
    def __init__(self, model: CaptionTransformer, token_encode_map, learning_rate: float):
        super().__init__()
        self.model = model
        self.encode_map = token_encode_map
        self.lr = learning_rate

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

    def training_step(self, batch, batch_idx):
        image, caption_sample, caption_target = batch
        logits = self(image, caption_sample)
        loss = self.loss_function(logits, caption_target)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, caption_sample, caption_target = batch
        logits = self(image, caption_sample)
        loss = self.loss_function(logits, caption_target)
        self.log("val_loss", loss)
        return loss
