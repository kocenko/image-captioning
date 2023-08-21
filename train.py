# TODO:
# > Training loop
# > Saving the model
# > Searching and loading the newest model
# > Keeping track of the hyperparameters along the training
# > Ensuring the possibility to restart training in any moment (without reloading the dataset etc.)
# > Saving loss function graphs
# > Saving the captions for one chosen image along the training process
# ...
# > Plotting attention maps
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from transformer import Decoder
from dataset import ImageCaptionDataset
from feature_extractor import FeatureExtractor
from tokenizer import Tokenizer


# Tokenizer does not change no matter what. It is completely dependent on the dataset.
# Feature extractor depends on the tokenizer attributes AND slicing point.
# Slicing point might be an interesting thing to parametrize and optimize.
# We can try implementing grid or random search looking for the best hyperparameters...
#
# Add also some metrics like BLEU or ROUGE...
class Trainer:
    allowed_optimizations: list[str] = ["grid", "random"]

    def __init__(self,
                 tokenizer: Tokenizer,
                 feature_extractor: FeatureExtractor,
                 dataset: ImageCaptionDataset,
                 hyperparams: dict):

        self.tokenizer: Tokenizer = tokenizer
        self.feature_extractor: FeatureExtractor = feature_extractor
        self.hyperparams: dict = hyperparams
        self.device: str = hyperparams["device"]

        # Splitting dataset into subsets
        generator = torch.Generator().manual_seed(42)
        split_lengths = hyperparams["split_lengths"]
        self.train_set, self.valid_set, self.test_set = random_split(dataset, split_lengths, generator=generator)

        # The main role
        self.decoder: Decoder = Decoder(**hyperparams)

    def __calc_single_loss(self, predictions, labels):
        B, T, C = predictions.shape
        logits = predictions.view(B * T, C)
        targets = labels.view(B * T)
        targets = targets.type(torch.LongTensor).to(self.device)
        loss = F.cross_entropy(logits, targets)

        return loss

    @torch.no_grad()
    def calculate_losses(self, iterations: int):
        split_type = ["train", "valid"]
        batch_size = self.hyperparams["batches"]
        outcome_losses = {}
        self.decoder.eval()
        for t, split in enumerate([self.train_set, self.valid_set]):
            loader = DataLoader(split, batch_size=batch_size, shuffle=True, drop_last=True)
            loader = iter(loader)
            losses = torch.zeros(iterations)
            for i in range(iterations):
                image, caption, label = loader.__next__()
                logits = self.decoder(image, caption)
                loss = self.__calc_single_loss(logits, label)
                losses[i] = loss.item()
            outcome_losses[split_type[t]] = losses.mean()
        self.decoder.train()
        return outcome_losses

    def train(self):
        lr = self.hyperparams["learning_rate"]
        epochs = self.hyperparams["epochs"]
        batch_size = self.hyperparams["batches"]
        eval_per_epoch = 10
        eval_iterations = 200
        break_iter = None

        if break_iter == None:
            break_iter = len(self.train_set) // batch_size

        train_dataloader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=lr)
        eval_each = break_iter // eval_per_epoch

        for e in range(epochs):
            for i, (x1, x2, y) in enumerate(train_dataloader):
                if i == break_iter:
                    break

                if i % eval_each == 0:
                    losses = self.calculate_losses(eval_iterations)
                    print(
                        f"Epoch: [{e + 1}/{epochs}] Step: [{i}/{break_iter}], train loss: {losses['train']:.4f}, val loss: {losses['valid']:.4f}")

                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                logits = self.decoder(x1, x2)
                loss = self.__calc_single_loss(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

