# TODO:
# > Saving loss function graphs
# > Saving the captions for one chosen image along the training process
# ...
# > Plotting attention maps
# Tokenizer does not change no matter what. It is completely dependent on the dataset.
# Feature extractor depends on the tokenizer attributes AND slicing point.
# Slicing point might be an interesting thing to parametrize and optimize.
# We can try implementing grid or random search looking for the best hyperparameters...
#
# Add also some metrics like BLEU or ROUGE...

import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from transformer import Decoder
from caption_generator import CaptionGenerator
from dataset import ImageCaptionDataset
from feature_extractor import FeatureExtractor
from tokenizer import Tokenizer


class Trainer:
    allowed_optimizations: list[str] = ["grid", "random"]
    datetime_format = "%Y-%m-%d %H-%M-%S"

    def __init__(self,
                 tokenizer: Tokenizer,
                 feature_extractor: FeatureExtractor,
                 dataset: ImageCaptionDataset,
                 checkpoint_path: str,
                 sample_image_path: str,
                 writer: SummaryWriter,
                 hyperparams: dict):

        self.tokenizer: Tokenizer = tokenizer
        self.feature_extractor: FeatureExtractor = feature_extractor
        self.checkpoint_path: str = checkpoint_path
        self.sample_image_path: str = sample_image_path
        self.writer: SummaryWriter = writer
        self.hyperparams: dict = hyperparams
        self.device: str = hyperparams["device"]

        # Splitting dataset into subsets
        generator = torch.Generator().manual_seed(42)
        split_lengths = hyperparams["split_lengths"]
        self.train_set, self.valid_set, self.test_set = random_split(dataset, split_lengths, generator=generator)

        # The main role
        self.decoder: Decoder = Decoder(**self.hyperparams)

    def __training_in_progress_path(self) -> Optional[str]:
        all_files = os.listdir(self.checkpoint_path)

        if len(all_files) == 0:
            return None

        # Finding all checkpoints
        epochs_done: Dict[str, Tuple[int, int, str]] = {}  # {Time, (epoch, epoch_max, path)
        for file in all_files:
            words = file.split('.')[0].split('_')

            if words[0] in epochs_done:
                if int(words[1]) > epochs_done[words[0]][0]:
                    epochs_done[words[0]] = (int(words[1]), int(words[3]), file)
            else:
                epochs_done[words[0]] = (int(words[1]), int(words[3]), file)

        # Choosing unfinished checkpoints
        timestamp_files = {}
        for checkpoint in epochs_done.keys():
            if epochs_done[checkpoint][0] < epochs_done[checkpoint][1]:
                timestamp = datetime.strptime(checkpoint, Trainer.datetime_format)
                timestamp_files[timestamp] = epochs_done[checkpoint][2]

        if len(timestamp_files) == 0:
            return None

        return timestamp_files[max(timestamp_files, key=timestamp_files.get)]

    def __calc_single_loss(self, predictions, labels):
        B, T, C = predictions.shape
        logits = predictions.view(B * T, C)
        targets = labels.view(B * T)
        targets = targets.type(torch.LongTensor).to(self.device)
        loss = F.cross_entropy(logits, targets)

        return loss

    def __on_epoch_end(self,
                       current_epoch: int,
                       number_of_epochs: int,
                       optimizer: torch.optim.AdamW,
                       progress_path: str):

        captioner = CaptionGenerator(self.decoder, self.tokenizer, self.feature_extractor, **self.hyperparams)
        self.writer.add_text("Captioner", captioner.generate(self.sample_image_path))

        e = current_epoch
        if progress_path is not None:
            path_to_save = progress_path.split('_')[0] + f"_{e + 1}_of_{number_of_epochs}.pt"
        else:
            path_to_save = datetime.now().strftime(Trainer.datetime_format) + f"_{e + 1}_of_{number_of_epochs}.pt"

        torch.save({
            'epoch': e,
            'model_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyperparams': self.hyperparams
        }, path_to_save)

    @torch.no_grad()
    def calculate_losses(self, iterations: int):
        split_type = ["train", "valid"]
        batch_size = self.hyperparams["batches"]
        outcome_losses = {}
        self.decoder.eval()
        for t, split in enumerate([self.train_set, self.valid_set]):
            loader = DataLoader(split, batch_size=batch_size, shuffle=True)
            loader = iter(loader)
            losses = torch.zeros(iterations)
            iterations = min(iterations, len(loader))
            for i in range(iterations):
                image, caption, label = loader.__next__()
                image, caption, label = image.to(self.device), caption.to(self.device), label.to(self.device)
                logits = self.decoder(image, caption).to(self.device)
                loss = self.__calc_single_loss(logits, label)
                losses[i] = loss.item()
            outcome_losses[split_type[t]] = losses.mean()
        self.decoder.train()
        return outcome_losses

    def train(self):
        progress_path = self.__training_in_progress_path()
        checkpoint = None
        current_epoch = 0

        if progress_path is not None:
            print(f"Loading progress from {progress_path}")
            checkpoint = torch.load(progress_path)
            self.hyperparams = checkpoint["hyperparams"]
            self.decoder.load_state_dict(checkpoint["model_state_dict"])
            current_epoch = checkpoint["epoch"]

        eval_iterations = self.hyperparams["eval_iterations"]
        eval_per_epoch = self.hyperparams["eval_per_epoch"]
        number_of_epochs = self.hyperparams["epochs"]
        batch_size = self.hyperparams["batches"]
        lr = self.hyperparams["learning_rate"]
        break_iter = len(self.train_set) // batch_size
        eval_each = break_iter // eval_per_epoch

        train_dataloader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, drop_last=True)
        optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=lr)

        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for e in range(current_epoch, number_of_epochs):
            for i, (x1, x2, y) in enumerate(train_dataloader):
                if i == break_iter:
                    break

                if i % eval_each == 0:
                    losses = self.calculate_losses(eval_iterations)
                    self.writer.add_scalar("train_loss", losses["train"], e * len(train_dataloader) + i)
                    self.writer.add_scalar("valid_loss", losses["valid"], e * len(train_dataloader) + i)
                    print(
                        f"Epoch: [{e + 1}/{number_of_epochs}] Step: [{i}/{break_iter}], train loss: {losses['train']:.4f}, val loss: {losses['valid']:.4f}")

                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                logits = self.decoder(x1, x2)
                loss = self.__calc_single_loss(logits, y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            self.__on_epoch_end(e, number_of_epochs, optimizer, progress_path)
