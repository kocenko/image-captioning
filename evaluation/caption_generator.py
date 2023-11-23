from typing import Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from model.transformer import CaptionTransformer
from data_processing.tokenizer import Tokenizer
from data_processing.image_transforms import ImageTransforms
from evaluation.extract_heads import extract_encoder_heads, extract_decoder_heads


@dataclass
class GeneratorCandidates:
    decoder_heads: Optional[list[torch.Tensor]]
    encoder_heads: Optional[list[list[torch.Tensor]]]
    indices: list[int]
    probability: float


class CaptionGenerator:
    """A class used for generating a caption from file using beam search

    Attributes:
        model (CaptionTransformer): decoder used for caption generation
        tokenizer (Tokenizer): custom tokenizer
        transform (ImageTransforms): transforms image
        bos (int): id of the beginning of sequence token
        eos (int): id of the end of sequence token
        vocab_size (int): number of tokens in vocabulary
        device (str): string indicating which device will be used for calculations
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
            device (str): string indicating which device will be used for calculations
        """

        self.model = model
        self.tokenizer = tokenizer
        self.transform = transform
        self.vocab_size = vocab_size
        self.bos = self.tokenizer.encode_map[Tokenizer.start_token]
        self.eos = self.tokenizer.encode_map[Tokenizer.end_token]
        self.device: str = device

    @staticmethod
    def topk_to_candidate(topk_output: torch.return_types.topk) -> list[GeneratorCandidates]:
        indices = topk_output.indices[0].tolist()
        values = topk_output.values[0].tolist()
        return [GeneratorCandidates(None, None, [ids], val) for ids, val in zip(indices, values)]

    def generate_beam_search(
        self, image_path: str, beam_width: int
    ) -> tuple[list[int], list[torch.Tensor], list[list[torch.Tensor]]]:
        image = self.transform.read_image(image_path).unsqueeze(0).to(self.device)
        image = self.transform.transform(image)
        caption_start = torch.tensor([self.bos], device=self.device).unsqueeze(0)

        self.model.eval()

        # Initial prediction
        logits = self.model(image, caption_start)[:, :, -1]
        predictions = F.softmax(logits, dim=-1)
        best = self.topk_to_candidate(torch.topk(predictions, beam_width, dim=-1))

        ready_captions = []
        captions_to_generate = beam_width
        while captions_to_generate > 0:
            decoder_heads = []
            encoder_heads = []
            all_probabilities = []
            for candidate in best:
                caption = torch.tensor([self.bos] + candidate.indices, device=self.device).unsqueeze(0)
                logits = self.model(image, caption)[:, :, -1]
                probabilities = F.softmax(logits, dim=-1) * candidate.probability
                all_probabilities.extend(probabilities)
                decoder_heads.append(extract_decoder_heads(self.model))
                encoder_heads.append(extract_encoder_heads(self.model))
            probabilities = torch.cat(all_probabilities, dim=-1).unsqueeze(0)
            new_best = self.topk_to_candidate(torch.topk(probabilities, captions_to_generate, dim=-1))

            to_remove = []
            for i, candidate in enumerate(new_best):
                parent_id = candidate.indices[0] // self.vocab_size
                child_id = candidate.indices[0] % self.vocab_size
                candidate.indices = best[parent_id].indices + [child_id]
                candidate.decoder_heads = decoder_heads[parent_id]
                candidate.encoder_heads = encoder_heads[parent_id]
                if child_id == self.eos or len(candidate.indices) == self.tokenizer.max_length - 1:
                    ready_captions.append(candidate)
                    to_remove.append(i)
                    captions_to_generate -= 1

            best = [c for i, c in enumerate(new_best) if i not in to_remove]

        self.model.train()

        best_caption = max(ready_captions, key=lambda x: x.probability)
        return (
            best_caption.indices,
            best_caption.encoder_heads,
            best_caption.decoder_heads,
        )

    def generate(self, image_path: str, max_size: int, temperature: float = 0.5) -> list[int]:
        """Method used to generate a caption

        Args:
            image_path (str): path to the image to generate the caption for
            max_size (int): maximal size of the caption
            temperature (float): scaling model's output to achieve different results

        Returns:
            String with the generated caption
        """
        max_size = min(max_size, self.tokenizer.max_length)
        generated_caption = torch.tensor([self.bos], device=self.device).unsqueeze(0)
        image = self.transform.read_image(image_path).unsqueeze(0).to(self.device)
        image = self.transform.transform(image)

        self.model.eval()
        for _ in range(max_size):
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

        self.model.train()
        caption_list = generated_caption[0].tolist()
        return caption_list


if __name__ == "__main__":
    pass
