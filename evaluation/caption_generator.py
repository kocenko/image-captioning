import torch
import torch.nn.functional as F
from torchvision.io import read_image

from model.transformer import CaptionTransformer
from data_processing.tokenizer import Tokenizer


class CaptionGenerator:
    """A class used for generating a caption from file

    Attributes:
        tokenizer (Tokenizer): custom tokenizer
        model (CaptionTransformer): decoder used for caption generation
        device (str): string indicating which device will be used for calculations
    """

    def __init__(self, model: CaptionTransformer, tokenizer: Tokenizer, device: str) -> None:
        """Initializes caption generator

        Args:
            tokenizer (Tokenizer): custom tokenizer
            model (CaptionTransformer): model used for caption generation
            device (str): string indicating which device will be used for calculations
        """
        self.tokenizer = tokenizer
        self.model = model
        self.device: str = device

    def generate(self, image_path: str, max_size: int, temperature: float = 0.5) -> str:
        """Method used to generate a caption

        Args:
            image_path (str): path to the image to generate the caption for
            max_size (int): maximal size of the caption
            temperature (float): scaling model's output to achieve different results

        Returns:
            String with the generated caption
        """
        max_size = min(max_size, self.tokenizer.max_length)
        generated_caption = [self.tokenizer.encode_map[self.tokenizer.start_token]]
        generated_caption = torch.tensor(generated_caption, device=self.device).unsqueeze(0)

        image = read_image(image_path).unsqueeze(0)

        self.model.eval()
        for _ in range(max_size):
            logits = self.model(image, generated_caption)
            logits = logits[:, -1, :]  # Fetching the last token of the generated sequence
            predictions = F.log_softmax(logits, dim=-1)
            if temperature == 0:
                new_token = torch.argmax(predictions, dim=-1).unsqueeze(0)
            else:
                new_token = torch.multinomial(predictions/temperature, num_samples=1)

            generated_caption = torch.cat([generated_caption, new_token], dim=1)
            if new_token[0].tolist() == [self.tokenizer.encode_map[Tokenizer.end_token]]:
                break

        self.model.train()
        caption_list = generated_caption[0].tolist()
        return self.tokenizer.decode(caption_list)


if __name__ == "__main__":
    pass
