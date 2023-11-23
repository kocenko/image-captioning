import torch
from model.transformer import CaptionTransformer


def extract_encoder_heads(model: CaptionTransformer) -> list[list[torch.Tensor]]:
    """Returns list of attention maps per head for self attention
    Expected output shape: list_of_layers[list_of_heads[attention_weight_shape]]
    """

    encoder_heads = [
        [head.cpu().detach() for head in block.attention.attention_weights[0]] for block in model.encoder_blocks
    ]

    return encoder_heads


def extract_decoder_heads(model: CaptionTransformer) -> list[torch.Tensor]:
    """Returns list of attention maps per head for last layer of decoder's cross attention
    Expected output shape: list_of_heads[attention_weight_shape]
    """

    decoder_heads = [head.cpu().detach() for head in model.decoder_blocks[-1].cross_attention.attention_weights[0]]
    return decoder_heads


def aggregate_heads(heads: list[torch.Tensor], method: str = "sum") -> torch.Tensor:
    assert method in ["sum", "mean"], f"Unsupported aggregation method {method}"
    stacked_heads = torch.stack(heads, dim=0)

    if method == "sum":
        return stacked_heads.sum(dim=0)
    if method == "mean":
        return stacked_heads.mean(dim=0)
