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
        image_size = config["image_size"]
        decoder_layers = config["decoder_layers"]
        cross_attention_key_dim = config["cross_att_key_dim"]

        self.image_transform = image_transform
        self.feature_extractor = feature_extractor
        if not self.feature_extractor:
            patch_size = config["patch_size"]
            shift_pixels = config["shift_pixels"]
            encoder_layers = config["encoder_layers"]
            patches_num = (image_size // patch_size) ** 2
            self.encoder_input = EncoderInput(image_size, shift_pixels, patch_size, embeddings)
            self.encoder_blocks = nn.Sequential(
                *[EncoderBlock(embeddings, dropout_rate, heads_num, patches_num) for _ in range(encoder_layers)]
            )

        self.decoder_input = DecoderInput(vocabulary_size, max_caption_length, embeddings)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(embeddings, cross_attention_key_dim, dropout_rate, heads_num, max_caption_length)
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

        if not self.feature_extractor:
            image = self.encoder_input(image)
            image = self.encoder_blocks(image)
        else:
            image = self.feature_extractor.feed(image)
            image = torch.flatten(image, start_dim=2)
            image = image.permute(0, 2, 1)

        for block in self.decoder_blocks:
            x = block(image, x, padding_mask)

        predictions = self.output_layer(x).contiguous()
        return predictions

    def load_weights(self, path_to_weights: str) -> None:
        with open("configs/custom_encoder_weights_names.txt", "r") as f:
            custom_names = f.read()
            custom_names = custom_names.splitlines()

        with open("configs/vit_encoder_weights_names.txt", "r") as f:
            vit_names = f.read()
            vit_names = vit_names.splitlines()

        mappings = {custom_name: vit_name for vit_name, custom_name in zip(vit_names, custom_names)}

        with torch.no_grad():
            vit_weights = torch.load(path_to_weights)

            matching_params = [
                (self_name, self_param) for self_name, self_param in self.named_parameters() if self_name in mappings
            ]

            for name, param in matching_params:
                pretrained = vit_weights[mappings[name]]

                # A hack used to remove 'cls' token from positional embedding
                if name == "encoder_input.positional_embedding.weight":
                    pretrained = pretrained[:, 1:, :].reshape(param.shape)

                param.data.copy_(pretrained)

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
