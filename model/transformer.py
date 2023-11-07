import torch
import torch.nn as nn

from model.encoder_input import EncoderInput
from model.encoder_block import EncoderBlock
from model.decoder_input import DecoderInput
from model.decoder_block import DecoderBlock
from model.decoder_output import DecoderOutput


class CaptionTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        embeddings = config["embeddings"]
        dropout_rate = config["dropout_rate"]
        heads_num = config["heads_num"]
        patch_size = config["patch_size"]
        image_size = config["image_size"]
        max_caption_length = config["max_caption_length"]
        vocabulary_size = config["vocabulary_size"]
        counter = config["counter"]
        encode_map = config["encode_map"]
        banned_tokens = config["banned_tokens"]
        shift_pixels = config["shift_pixels"]
        encoder_layers = config["encoder_layers"]
        decoder_layers = config["decoder_layers"]
        device = config["device"]
        patches_num = (image_size[0] // patch_size) * (image_size[1] // patch_size)

        encoder_blocks = nn.Sequential(
            *[EncoderBlock(embeddings, dropout_rate, heads_num, patches_num, device) for _ in range(encoder_layers)]
        )
        self.encoder = nn.Sequential(
            EncoderInput(image_size, shift_pixels, patch_size, embeddings, device),
            encoder_blocks,
        )

        self.decoder_input = DecoderInput(vocabulary_size, max_caption_length, embeddings, device)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(embeddings, dropout_rate, heads_num, max_caption_length, device)
                for _ in range(decoder_layers)
            ]
        )
        self.output_layer = DecoderOutput(embeddings, vocabulary_size, device, True, counter, encode_map, banned_tokens)

    def forward(self, image: torch.Tensor, caption: torch.Tensor):
        image_embeddings = self.encoder(image)
        x = self.decoder_input(caption)

        for block in self.decoder_blocks:
            x = block(image_embeddings, x)

        predictions = self.output_layer(x)
        return predictions
