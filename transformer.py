import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.flattener = nn.Flatten(start_dim=2, end_dim=3)

    def forward(self, x):
        # Expected input (B, C, H, W)
        return self.flattener(x)


class SingleHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()

        # Self attention

        # Cross attention

        # Feed forward


class TransformerDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Embeddings (with positional)


