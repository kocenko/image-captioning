from typing import Optional, Union
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


from model.encoder import EncoderInput
from model.encoder import EncoderBlock
from model.decoder import DecoderInput
from model.decoder import DecoderBlock
from model.decoder import DecoderOutput
from evaluation.beam_search_util import CandidateNode, CandidateGraph
from data_processing.feature_extractor import FeatureExtractor
from data_processing.tokenizer import Tokenizer
from data_processing.image_transforms import ImageTransforms


class CaptionTransformer(nn.Module):
    def __init__(
        self,
        tokenizer: Tokenizer,
        image_transform: ImageTransforms,
        feature_extractor: Optional[FeatureExtractor] = None,
        **config
    ):
        super().__init__()
        self.tokenizer = tokenizer
        counter = self.tokenizer.counter
        encode_map = self.tokenizer.encode_map
        max_caption_length = config["max_caption_length"]
        vocabulary_size = config["vocabulary_size"]
        banned_tokens = config["banned_tokens"]

        embeddings = config["embeddings"]
        dropout_rate = config["dropout_rate"]
        heads_num = config["heads_num"]
        decoder_layers = config["decoder_layers"]
        encoder_embeddings = config["encoder_embeddings"]

        self.image_transform = image_transform
        self.feature_extractor = feature_extractor

        self.decoder_input = DecoderInput(vocabulary_size, max_caption_length, embeddings)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(embeddings, encoder_embeddings, dropout_rate, heads_num, max_caption_length)
                for _ in range(decoder_layers)
            ]
        )
        self.output_layer = DecoderOutput(embeddings, vocabulary_size, True, counter, encode_map, banned_tokens)
        self.register_buffer(
            "start_caption", torch.tensor([encode_map[Tokenizer.start_token]]).unsqueeze(0), persistent=False
        )

    @staticmethod
    def get_padding_mask(caption_batch: torch.Tensor, padding_idx: int = 0) -> torch.Tensor:
        # noinspection PyTypeChecker
        return caption_batch == padding_idx

    def forward(self, image: torch.Tensor, caption: torch.Tensor):
        padding_mask = self.get_padding_mask(caption)
        x = self.decoder_input(caption, padding_mask)

        image = self.feature_extractor.feed(image)
        if len(image.shape) > 3:
            image = torch.flatten(image, start_dim=2)
            image = image.permute(0, 2, 1)

        for block in self.decoder_blocks:
            x = block(image, x, padding_mask)

        predictions = self.output_layer(x).contiguous()
        return predictions

    @staticmethod
    def find_top_best(
        probabilities: torch.Tensor, k: int, beam_width: int, root_node: CandidateNode, search_graph: CandidateGraph
    ) -> list[CandidateNode]:
        children = []
        topk = torch.topk(probabilities, k)
        for i, (token_id, probability) in enumerate(zip(topk.indices.tolist(), topk.values.tolist())):
            child = CandidateNode(
                root_node.id * beam_width + (i + 1),
                root_node.tokens + [token_id],
                root_node.probability * probability,
                False,
                False,
            )
            search_graph.nodes[child.id] = child
            search_graph.edges[root_node.id].append(child.id)
            children.append(child)
        return children

    @torch.no_grad()
    def generate_beam_search(
        self, image: Union[str, torch.Tensor], beam_width: int
    ) -> tuple[list[int], CandidateGraph]:
        bos = self.tokenizer.encode_map[Tokenizer.start_token]
        eos = self.tokenizer.encode_map[Tokenizer.end_token]

        self.eval()
        if type(image) is str:
            image = self.image_transform.transform(self.image_transform.read_image(image).unsqueeze(0))

        # Initialization
        search_graph = CandidateGraph({}, defaultdict(list))
        root_node = CandidateNode(0, [bos], 1.0, True, False)
        search_graph.nodes[0] = root_node
        best_nodes = [root_node]

        ready_captions = []
        to_generate = beam_width
        while to_generate > 0:
            all_best = []
            for candidate in best_nodes:
                candidate.best = True
                if candidate.tokens[-1] == eos or len(candidate.tokens) == self.tokenizer.max_length - 1:
                    candidate.last = True
                    ready_captions.append(candidate)
                    to_generate -= 1

            for candidate in [node for node in best_nodes if node not in ready_captions]:
                caption = torch.tensor(
                    candidate.tokens, device="cuda" if image.get_device() != -1 else "cpu"
                ).unsqueeze(0)
                logits = self(image, caption)[:, :, -1]
                probabilities = F.softmax(logits, dim=-1).squeeze()
                all_best.extend(self.find_top_best(probabilities, to_generate, beam_width, candidate, search_graph))

            best_nodes = sorted(all_best, key=lambda x: x.probability, reverse=True)[:to_generate]

        best_candidate = max(ready_captions, key=lambda x: x.probability)
        tokens_to_return = best_candidate.tokens

        self.train()
        return tokens_to_return, search_graph

    @torch.no_grad()
    def generate(self, image: Union[str, torch.Tensor], temperature: float = 0.5) -> list[int]:
        """Method used to generate a caption

        Args:
            image (str): path to the image to generate the caption for
            temperature (float): scaling model's output to achieve different results

        Returns:
            String with the generated caption
        """
        eos = self.tokenizer.encode_map[Tokenizer.end_token]
        self.eval()

        generated_caption = self.start_caption
        if type(image) is str:
            image = self.image_transform.transform(self.image_transform.read_image(image).unsqueeze(0))

        for _ in range(self.tokenizer.max_length - 1):
            logits = self(image, generated_caption)
            logits = logits[:, :, -1]  # Fetching the last token of the generated sequence
            predictions = F.softmax(logits, dim=-1)
            if temperature == 0:
                new_token = torch.argmax(predictions, dim=-1).unsqueeze(0)
            else:
                new_token = torch.multinomial(predictions / temperature, num_samples=1)

            generated_caption = torch.cat([generated_caption, new_token], dim=1)
            if new_token[0].tolist() == [eos]:
                break

        caption_list = generated_caption[0].tolist()

        self.train()
        return caption_list
