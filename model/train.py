import os
import tqdm
import random
from collections import defaultdict
from datetime import datetime
from itertools import islice
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_image

from model.transformer import CaptionTransformer
from evaluation.caption_generator import CaptionGenerator
from data_processing.dataset import ImageCaptionDataset
from data_processing.tokenizer import Tokenizer


class Trainer:
    """A class used for training a transformer

    Attributes:
        model (CaptionTransformer): model instance to train
        tokenizer (Tokenizer): custom tokenizer
        vocabulary_size (int): number of tokens in vocabulary
        checkpoint_path (str): path of the folder where checkpoint files are saved
        sample_image_path (str): path to the file, which is used to generate captions
        writer (SummaryWriter): log writer object
        hyperparams (dict): dict of parameters used in the training
        device (str): string indicating which device will be used for calculations
    """

    allowed_optimizations: list[str] = ["grid", "random"]
    datetime_format = "%Y-%m-%d %H-%M-%S"

    def __init__(
        self,
        model: CaptionTransformer,
        tokenizer: Tokenizer,
        vocabulary_size: int,
        datasets: list[ImageCaptionDataset],
        raw_test: list[tuple[str, str]],
        checkpoint_path: str,
        sample_image_path: str,
        writer: SummaryWriter,
        hyperparams: dict,
    ) -> None:
        """Initializes Trainer class

        Args:
            tokenizer (Tokenizer): custom tokenizer
            vocabulary_size (int): number of tokens in vocabulary
            datasets (list[ImageCaptionDataset]): train, valid and test datasets
            checkpoint_path (str): path of the folder where checkpoint files are saved
            sample_image_path (str): path to the file, which is used to generate captions
            writer (SummaryWriter): log writer object
            hyperparams (dict): dict of parameters used in the training
        """

        self.datasets = defaultdict()
        self.datasets["train"], self.datasets["valid"], self.datasets["test"] = datasets
        self.raw_test_dataset = raw_test
        self.model = model
        self.tokenizer = tokenizer
        self.vocabulary_size = vocabulary_size
        self.checkpoint_path = checkpoint_path
        self.sample_image_path = sample_image_path
        self.writer = writer
        self.hyperparams = hyperparams
        self.device = hyperparams["device"]
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.encode_map[Tokenizer.empty_token], reduction="none")

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
            words = file.split(".")[0].split("_")

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

    def __calc_single_loss(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculates a loss of a single predictions-labels pair

        Args:
            labels (Tensor): ground truth captions

        Returns:
            Tensor as an output of the cross entropy with logarithmic softmax. Calculated for the whole batch.
        """

        loss = self.criterion(predictions, labels)

        mask = (labels != self.tokenizer.encode_map[Tokenizer.empty_token]) & (loss < 1e8)
        mask = mask.float()

        loss = loss * mask
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

    @staticmethod
    def __calc_masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> torch.float32:
        logits = logits.transpose(-2, -1)
        mask = labels != 0
        predictions = torch.argmax(logits, dim=-1)
        labels = labels.to(torch.int64)
        match = (predictions == labels).to(mask.dtype)
        acc = torch.sum(match * mask) / torch.sum(mask)
        return acc

    def __on_epoch_end(
        self, current_epoch: int, number_of_epochs: int, optimizer: torch.optim.AdamW, progress_path: str
    ) -> None:
        """
        Adds log to the writer and saves the current state of the model

        Args:
            current_epoch (int): number of the current epoch
            number_of_epochs (int): number of all epochs
            optimizer (AdamW): optimizer (used to save its state)
            progress_path (str): path to the folder where the checkpoint is saved
        """
        e = current_epoch
        path_to_save = progress_path.split("_")[0] + f"_{e + 1}_of_{number_of_epochs}.pt"

        torch.save(
            {
                "epoch": e,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "hyperparams": self.hyperparams,
            },
            os.path.join(self.checkpoint_path, path_to_save),
        )

        exemplary_image, exemplary_caption = random.choice(self.raw_test_dataset)
        captioner = CaptionGenerator(self.model, self.tokenizer, self.vocabulary_size, self.device)
        generated_caption = captioner.generate_beam_search(exemplary_image, 3)
        self.writer.add_text("caption", generated_caption, e)

        image = read_image(exemplary_image).permute(1, 2, 0)
        fig, ax = plt.subplots(1)
        ax.imshow(image)
        bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
        ax.text(
            image.shape[1] // 2,
            0.02,
            f"Original: {exemplary_caption}",
            ha="center",
            va="center",
            size=10,
            bbox=bbox_props,
        )
        ax.text(
            image.shape[1] // 2,
            image.shape[0] + 0.02,
            f"Generated: {generated_caption[7:-5]}",
            ha="center",
            va="center",
            size=10,
            bbox=bbox_props,
        )
        plt.show()

    @torch.no_grad()
    def calculate_losses_and_accuracy(self, iterations: int, batch_size: int):
        outcome_losses = {}
        outcome_accuracy = {}

        self.model.eval()
        for split in ["train", "valid"]:
            losses = torch.zeros(iterations)
            accuracies = torch.zeros(iterations)
            loader = DataLoader(self.datasets[split], batch_size=batch_size, shuffle=True)

            for i, (image, caption, label) in enumerate(islice(loader, iterations)):
                image, caption, label = image.to(self.device), caption.to(self.device), label.to(self.device)
                logits = self.model(image, caption).to(self.device)
                loss = self.__calc_single_loss(logits, label)
                acc = self.__calc_masked_accuracy(logits, label)
                losses[i] = loss.item()
                accuracies[i] = acc.item()

            outcome_accuracy[split] = accuracies.mean()
            outcome_losses[split] = losses.mean()
        self.model.train()

        return outcome_losses, outcome_accuracy

    def train(self):
        progress_path = self.__training_in_progress_path()
        checkpoint = None
        current_epoch = 0

        if progress_path:
            print(f"Loading progress from {progress_path}")
            checkpoint = torch.load(os.path.join(self.checkpoint_path, progress_path))
            self.hyperparams = checkpoint["hyperparams"]
            self.model.load_state_dict(checkpoint["model_state_dict"])
            current_epoch = checkpoint["epoch"]
        else:
            progress_path = datetime.now().strftime(Trainer.datetime_format) + "_p"

        eval_iterations = self.hyperparams["eval_iterations"]
        eval_per_epoch = self.hyperparams["eval_per_epoch"]
        number_of_epochs = self.hyperparams["epochs"]
        batch_size = self.hyperparams["batches"]
        lr = self.hyperparams["learning_rate"]

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        if checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        all_iters = len(self.datasets["train"]) // batch_size
        eval_each = all_iters // min(all_iters, eval_per_epoch)

        for e in range(current_epoch, number_of_epochs):
            print(f"Epoch {e + 1}/{number_of_epochs}")
            train_dataloader = DataLoader(self.datasets["train"], batch_size=batch_size, shuffle=True)

            for i, (x1, x2, y) in (loading_bar := tqdm.tqdm(enumerate(train_dataloader), colour="00ff00")):
                if i % eval_each == 0:
                    losses, accuracy = self.calculate_losses_and_accuracy(eval_iterations, batch_size)
                    loading_bar.set_postfix_str(
                        f"Train loss: {losses['train']:.4f} --- "
                        f"Train accuracy {accuracy['train']:.4f} --- "
                        f"Validation loss: {losses['valid']:.4f} --- "
                        f"Validation accuracy {accuracy['valid']:.4f}"
                    )

                    self.writer.add_scalar("train_loss", losses["train"], e * all_iters + i)
                    self.writer.add_scalar("valid_loss", losses["valid"], e * all_iters + i)
                    self.writer.add_scalar("train_acc", accuracy["train"], e * all_iters + i)
                    self.writer.add_scalar("valid_acc", accuracy["valid"], e * all_iters + i)

                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model(x1, x2)
                loss = self.__calc_single_loss(logits, y)
                loss.backward()
                optimizer.step()

            self.__on_epoch_end(e, number_of_epochs, optimizer, progress_path)
