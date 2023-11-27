from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model.transformer import CaptionTransformer
from data_processing.tokenizer import Tokenizer
from data_processing.image_transforms import ImageTransforms


@dataclass
class CandidateNode:
    tokens: list[int]
    probability: float
    parent: Optional[object]
    children: Optional[list[object]]
    best: bool


class CaptionGenerator:
    """A class used for generating a caption from file using beam search

    Attributes:
        model (CaptionTransformer): decoder used for caption generation
        tokenizer (Tokenizer): custom tokenizer
        transform (ImageTransforms): transforms image
        bos (int): id of the beginning of sequence token
        eos (int): id of the end of sequence token
        vocab_size (int): number of tokens in vocabulary
    """

    def __init__(
        self, model: CaptionTransformer, tokenizer: Tokenizer, transform: ImageTransforms, vocab_size: int, device: str
    ) -> None:
        """Initializes caption generator

        Args:
            model (CaptionTransformer): model used for caption generation
            tokenizer (Tokenizer): custom tokenizer
            transform (ImageTransforms): transforms image
            vocab_size (int): number of tokens in vocabulary
        """

        self.model = model
        self.tokenizer = tokenizer
        self.transform = transform
        self.vocab_size = vocab_size
        self.bos = self.tokenizer.encode_map[Tokenizer.start_token]
        self.eos = self.tokenizer.encode_map[Tokenizer.end_token]
        self.device = device

    @staticmethod
    def find_top_best(probabilities: torch.Tensor, k: int, root_node: CandidateNode) -> list[CandidateNode]:
        children = []
        topk = torch.topk(probabilities, k)
        for token_id, probability in zip(topk.indices.tolist(), topk.values.tolist()):
            children.append(
                CandidateNode(
                    root_node.tokens + [token_id], root_node.probability * probability, root_node, None, False
                )
            )
        root_node.children = children
        return children

    @torch.no_grad()
    def generate_beam_search(self, image_path: str, beam_width: int) -> tuple[list[int], CandidateNode]:
        self.model.eval()

        image = self.transform.transform(self.transform.read_image(image_path).unsqueeze(0))
        caption_start = torch.tensor([self.bos], device=self.device).unsqueeze(0)

        # Initial prediction
        root_node = CandidateNode([self.bos], 1.0, None, None, True)
        logits = self.model(image, caption_start)[:, :, -1]
        predictions = F.softmax(logits, dim=-1).squeeze()
        best_nodes = self.find_top_best(predictions, beam_width, root_node)

        ready_captions = []
        while beam_width > 0:
            all_best = []
            for candidate in best_nodes:
                candidate.best = True
                if candidate.tokens[-1] == self.eos or len(candidate.tokens) == self.tokenizer.max_length - 1:
                    ready_captions.append(candidate)
                    beam_width -= 1
                    continue
                caption = torch.tensor(candidate.tokens).unsqueeze(0)
                logits = self.model(image, caption)[:, :, -1]
                probabilities = F.softmax(logits, dim=-1).squeeze()
                all_best.extend(self.find_top_best(probabilities, beam_width, candidate))

            if beam_width > 0:
                best_nodes = sorted(all_best, key=lambda x: x.probability, reverse=True)[:beam_width]

        best_candidate = max(ready_captions, key=lambda x: x.probability)
        tokens_to_return = best_candidate.tokens

        self.model.train()
        return tokens_to_return, root_node

    @torch.no_grad()
    def generate(self, image_path: str, temperature: float = 0.5) -> list[int]:
        """Method used to generate a caption

        Args:
            image_path (str): path to the image to generate the caption for
            temperature (float): scaling model's output to achieve different results

        Returns:
            String with the generated caption
        """

        self.model.eval()

        generated_caption = torch.tensor([self.bos], device=self.device).unsqueeze(0)
        image = self.transform.read_image(image_path).unsqueeze(0)
        image = self.transform.transform(image)
        if torch.cuda.is_available():
            image = image.cuda()

        for _ in range(self.tokenizer.max_length - 1):
            logits = self.model(image, generated_caption)
            logits = logits[:, :, -1]  # Fetching the last token of the generated sequence
            predictions = F.softmax(logits, dim=-1)
            if temperature == 0:
                new_token = torch.argmax(predictions, dim=-1).unsqueeze(0)
            else:
                new_token = torch.multinomial(predictions / temperature, num_samples=1)

            generated_caption = torch.cat([generated_caption, new_token], dim=1)
            if new_token[0].tolist() == [self.eos]:
                break

        caption_list = generated_caption[0].tolist()

        self.model.train()
        return caption_list


if __name__ == "__main__":
    pass
