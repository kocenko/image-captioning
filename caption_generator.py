import torch
import torch.nn.functional as F
from transformer import Decoder
from tokenizer import Tokenizer


class CaptionGenerator:
    def __init__(self, decoder: Decoder, tokenizer: Tokenizer, **kwargs):
        self.tokenizer: Tokenizer = tokenizer
        self.config = kwargs
        self.decoder: Decoder = decoder

    def generate(self, image: torch.Tensor, max_size: int) -> str:
        start_vector = self.tokenizer.encode(self.tokenizer.START_TOKEN).unsqueeze(0)
        start_vector = start_vector[:, :1]
        image = image.unsqueeze(0)

        while True:
            if start_vector.shape[1] == max_size - 1:
                new_token = self.tokenizer.encode(self.tokenizer.END_TOKEN)[:1].unsqueeze(0)
            else:
                logits = self.decoder(image, start_vector)
                logits = logits[:, -1, :]
                predictions = F.softmax(logits, dim=-1)
                new_token = torch.multinomial(predictions, num_samples=1)

            start_vector = torch.cat([start_vector, new_token], dim=1)

            if new_token == self.tokenizer.encode(self.tokenizer.END_TOKEN)[:1]:
                break

        return self.tokenizer.decode(start_vector[0, 1:-1].detach().numpy().tolist())
