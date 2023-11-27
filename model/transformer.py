from typing import Optional

import torch
import torch.nn as nn

from model.encoder import EncoderInput
from model.encoder import EncoderBlock
from model.decoder import DecoderInput
from model.decoder import DecoderBlock
from model.decoder import DecoderOutput
from data_processing.feature_extractor import FeatureExtractor


class CaptionTransformer(nn.Module):
    def __init__(self, feature_extractor: Optional[FeatureExtractor] = None, **config):
        super().__init__()
        embeddings = config["embeddings"]
        dropout_rate = config["dropout_rate"]
        heads_num = config["heads_num"]
        image_size = config["image_size"]
        max_caption_length = config["max_caption_length"]
        vocabulary_size = config["vocabulary_size"]
        counter = config["counter"]
        encode_map = config["encode_map"]
        banned_tokens = config["banned_tokens"]
        decoder_layers = config["decoder_layers"]
        cross_attention_key_dim = config["cross_att_key_dim"]

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
        with open('configs/custom_encoder_weights_names.txt', 'r') as f:
            custom_names = f.read()
            custom_names = custom_names.splitlines()

        with open('configs/vit_encoder_weights_names.txt', 'r') as f:
            vit_names = f.read()
            vit_names = vit_names.splitlines()

        mappings = {custom_name: vit_name for vit_name, custom_name in zip(vit_names, custom_names)}

        with torch.no_grad():
            vit_weights = torch.load(path_to_weights)

            matching_params = [
                (self_name, self_param)
                for self_name, self_param in self.named_parameters()
                if self_name in mappings and self_param.shape == vit_weights[mappings[self_name]].shape
            ]

            for name, param in matching_params:
                pretrained = vit_weights[mappings[name]]
                param.data.copy_(pretrained)
