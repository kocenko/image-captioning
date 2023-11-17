import torch
from torch import nn


class ResidualLayerNormalization(nn.Module):
    """Transformer layer performing residual connection with layer normalization

    Args:
        embeddings_number (int): embeddings dimension size
        device (int): on which device the layer normalization will be performed

    Attributes:
        layer_normalization (nn.LayerNorm): layer normalization module

    Methods:
        forward: Calculates the layer normalization on the sum of inputs of residual connection
    """

    def __init__(self, embeddings_number: int, device: str):
        super().__init__()
        self.layer_normalization = nn.LayerNorm(embeddings_number, device=device)

    def forward(self, x, residual) -> torch.Tensor:
        assert (
            x.shape == residual.shape
        ), f"Shapes of the input and residual do not match: input: {x.shape}, residual: {residual.shape}."

        return self.layer_normalization(x + residual)
