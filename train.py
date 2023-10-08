import os
from datetime import datetime
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter

from transformer import Decoder
from caption_generator import CaptionGenerator
from dataset import ImageCaptionDataset
from feature_extractor import FeatureExtractor
from tokenizer import Tokenizer


class Trainer:
    """
    A class used for training the decoder

    Attributes:
        tokenizer (Tokenizer): custom tokenizer
        feature_extractor (FeatureExtractor): pre-trained feature extractor
        checkpoint_path (str): path of the folder where checkpoint files are saved
        sample_image_path (str): path to the file, which is used to generate captions
        writer (SummaryWriter): log writer object
        hyperparams (dict): dict of parameters used in the training
        train_set (Subset): subset of the dataset used for training
        valid_set (Subset): subset of the dataset used for validation
        test_set (Subset): subset of the dataset used for testing
        decoder (Decoder): decoder used for caption generation
        device (str): string indicating which device will be used for calculations
        test (bool): whether to load only one sample for each subset of the dataset
    """

    allowed_optimizations: list[str] = ["grid", "random"]
    datetime_format = "%Y-%m-%d %H-%M-%S"

    def __init__(
            self,
            tokenizer: Tokenizer,
            feature_extractor: FeatureExtractor,
            dataset: ImageCaptionDataset,
            checkpoint_path: str,
            sample_image_path: str,
            writer: SummaryWriter,
            hyperparams: dict,
            test: bool = False
    ) -> None:
        """
        Initializes Trainer class

        Args:
            tokenizer (Tokenizer): custom tokenizer
            feature_extractor (FeatureExtractor): pre-trained feature extractor
            dataset (ImageCaptionDataset): custom dataset
            checkpoint_path (str): path of the folder where checkpoint files are saved
            sample_image_path (str): path to the file, which is used to generate captions
            writer (SummaryWriter): log writer object
            hyperparams (dict): dict of parameters used in the training
            test (bool): whether to load only one sample for each subset of the dataset
        """

        self.tokenizer: Tokenizer = tokenizer
        self.feature_extractor: FeatureExtractor = feature_extractor
        self.checkpoint_path: str = checkpoint_path
        self.sample_image_path: str = sample_image_path
        self.writer: SummaryWriter = writer
        self.hyperparams: dict = hyperparams
        self.device: str = hyperparams["device"]
        self.test: bool = test

        # Splitting dataset into subsets
        if not test:
            generator = torch.Generator().manual_seed(42)
            split_lengths = hyperparams["split_lengths"]
            self.train_set, self.valid_set, self.test_set = random_split(dataset, split_lengths, generator=generator)
        else:
            self.train_set, self.valid_set, self.test_set = Subset(dataset, [0]), Subset(dataset, [1]), Subset(dataset, [2])

        self.decoder: Decoder = Decoder(**self.hyperparams)

    def __training_in_progress_path(self) -> Optional[str]:
        """
        A method used for finding path to the checkpoint of the training in progress

        It returns a path to the file with a checkpoint. Checkpoint contains model's and optimizer's parameters and
        other saved parameters. The name of the file indicates whether the training is finished (the number of
        the current epoch is equal to the number of all epochs). When there are multiple files of the ongoing trainings,
        it returns the one which started the latest.

        Returns:
            None if there is not any checkpoint file of ongoing training
            Path to the file (string) if there are files in the folder of the model in training
        """

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

    @staticmethod
    def __calc_single_loss(predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates a loss of a single predictions-labels pair

        Args:
            predictions (Tensor): captions as an output from the decoder (as logits)
            labels (Tensor): ground truth captions

        Returns:
            Tensor as an output of the cross entropy with logarithmic softmax. Calculated for the whole batch.
        """
        b, t, c = predictions.shape
        predictions = predictions.view(b * t, c)
        labels = labels.view(b * t)

        loss = F.cross_entropy(predictions, labels, reduction='none')

        mask = (labels != 0) & (loss < 1e8)
        mask = mask.float()

        loss = loss * mask
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

    @staticmethod
    def __calc_masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.float32:
        mask = (labels != 0)
        predictions = torch.argmax(logits, dim=-1)
        labels = labels.to(torch.int64)
        match = (predictions == labels).to(mask.dtype)
        acc = torch.sum(match * mask) / torch.sum(mask)
        return acc

    def __on_epoch_end(self,
                       current_epoch: int,
                       number_of_epochs: int,
                       optimizer: torch.optim.AdamW,
                       progress_path: str
                       ) -> None:
        """
        Adds log to the writer and saves the current state of the model

        Args:
            current_epoch (int): number of the current epoch
            number_of_epochs (int): number of all epochs
            optimizer (AdamW): optimizer (used to save its state)
            progress_path (str): path to the folder where the checkpoint is saved
        """

        device = self.hyperparams["device"]
        captioner = CaptionGenerator(self.decoder, self.tokenizer, self.feature_extractor, device)
        self.writer.add_text("Captioner", captioner.generate(self.sample_image_path, max_size=80))

        e = current_epoch
        path_to_save = progress_path.split('_')[0] + f"_{e + 1}_of_{number_of_epochs}.pt"

        path_to_save = os.path.join(self.checkpoint_path, path_to_save)
        torch.save({
            'epoch': e,
            'model_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'hyperparams': self.hyperparams
        }, path_to_save)

    @torch.no_grad()
    def calculate_losses_and_accuracy(self, iterations: int):
        split_type = ["train", "valid"]
        batch_size = self.hyperparams["batches"]
        outcome_losses = {}
        outcome_accuracy = {}
        self.decoder.eval()
        for t, split in enumerate([self.train_set, self.valid_set]):
            loader = DataLoader(split, batch_size=batch_size, shuffle=True)
            loader = iter(loader)
            losses = torch.zeros(iterations)
            accuracies = torch.zeros(iterations)
            iterations = min(iterations, len(loader))
            for i in range(iterations):
                image, caption, label = loader.__next__()
                image, caption, label = image.to(self.device), caption.to(self.device), label.to(self.device)
                logits = self.decoder(image, caption).to(self.device)
                loss = self.__calc_single_loss(logits, label)
                acc = self.__calc_masked_accuracy(logits, label)

                losses[i] = loss.item()
                accuracies[i] = acc.item()

            outcome_accuracy[split_type[t]] = accuracies.mean()
            outcome_losses[split_type[t]] = losses.mean()
        self.decoder.train()
        return outcome_losses, outcome_accuracy

    def train(self):
        progress_path = self.__training_in_progress_path()
        checkpoint = None
        current_epoch = 0

        if progress_path is not None:
            print(f"Loading progress from {progress_path}")
            checkpoint = torch.load(os.path.join(self.checkpoint_path, progress_path))
            self.hyperparams = checkpoint["hyperparams"]
            self.decoder.load_state_dict(checkpoint["model_state_dict"])
            current_epoch = checkpoint["epoch"]
        else:
            progress_path = datetime.now().strftime(Trainer.datetime_format) + "_p"

        eval_iterations = self.hyperparams["eval_iterations"]
        eval_per_epoch = self.hyperparams["eval_per_epoch"]
        number_of_epochs = self.hyperparams["epochs"]
        batch_size = self.hyperparams["batches"]
        lr = self.hyperparams["learning_rate"]

        train_dataloader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True, drop_last=(not self.test))
        optimizer = torch.optim.AdamW(self.decoder.parameters(), lr=lr)

        if checkpoint is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        all_iters = len(train_dataloader)
        eval_each = all_iters // min(all_iters, eval_per_epoch)

        for e in range(current_epoch, number_of_epochs):
            for i, (x1, x2, y) in enumerate(train_dataloader):
                if i % eval_each == 0:
                    losses, accuracy = self.calculate_losses_and_accuracy(eval_iterations)

                    print(f"Epoch: [{e + 1}/{number_of_epochs}], "
                          f"Step: [{i}/{all_iters}], "
                          f"Train loss: {losses['train']:.4f}, "
                          f"Val loss: {losses['valid']:.4f}, "
                          f"Train acc: {accuracy['train']:.4f}, "
                          f"Val acc: {accuracy['valid']:.4f}")

                    self.writer.add_scalar("train_loss", losses["train"], e * len(train_dataloader) + i)
                    self.writer.add_scalar("valid_loss", losses["valid"], e * len(train_dataloader) + i)

                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.decoder(x1, x2)
                loss = self.__calc_single_loss(logits, y)
                loss.backward()
                optimizer.step()

            self.__on_epoch_end(e, number_of_epochs, optimizer, progress_path)
