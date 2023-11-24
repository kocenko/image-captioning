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

    def forward(self, image: torch.Tensor, caption: torch.Tensor):
        if not self.feature_extractor:
            image = self.encoder_input(image)
            image = self.encoder_blocks(image)
        else:
            image = self.feature_extractor.feed(image)
            image = torch.flatten(image, start_dim=2)
            image = image.permute(0, 2, 1)

        x, padding_mask = self.decoder_input(caption)
        for block in self.decoder_blocks:
            x = block(image, x, padding_mask)

        predictions = self.output_layer(x).contiguous()
        return predictions

    def load_weights(self, path_to_weights: str) -> None:
        mapping = {
            "embeddings.position_embeddings": "0.positional_embedding.weight",
            "embeddings.patch_embeddings.projection": "0.patch_embedding",
            "attention.attention.query": "self_attention.query_projection",
            "attention.attention.key": "self_attention.key_projection",
            "attention.attention.value": "self_attention.value_projection",
            "attention.output.dense": "self_attention.output_projection",
            "intermediate.dense": "feed_forward.0",
            "output.dense": "feed_forward.2",
            "layernorm_before": "add_and_norm_1.layer_normalization",
            "layernorm_after": "add_and_norm_2.layer_normalization",
            "encoder.layer": "1",
            "vit": "encoder",
        }

        def transform_name(old_name: str):
            for pretrained, custom in mapping.items():
                old_name = old_name.replace(pretrained, custom)
            return old_name

        with torch.no_grad():
            all_weights = torch.load(path_to_weights)
            for name, val in all_weights.items():
                new_name = transform_name(name)
                matching_params = [
                    param for name, param in self.named_parameters() if name == new_name and param.shape == val.shape
                ]
                for param in matching_params:
                    param.data.copy_(val)
