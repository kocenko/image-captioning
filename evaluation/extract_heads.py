import torch
from model.transformer import CaptionTransformer


def extract_encoder_heads(model: CaptionTransformer) -> list[list[torch.Tensor]]:
    """Returns list of attention maps per head for self attention
    Expected output shape: list_of_layers[list_of_heads[attention_weight_shape]]
    """

    encoder_heads = [
        [head.cpu().detach() for head in block.attention.attention_weights[0]]
        for block in model.encoder_blocks
    ]

    return encoder_heads


def extract_decoder_heads(model: CaptionTransformer) -> list[list[torch.Tensor]]:
    """Returns list of attention maps per head for cross attention
    Expected output shape: list_of_layers[list_of_heads[attention_weight_shape]]
    """

    decoder_heads = [
        [head.cpu().detach() for head in block.cross_attention.attention_weights[0]]
        for block in model.encoder_blocks
    ]

    return decoder_heads
