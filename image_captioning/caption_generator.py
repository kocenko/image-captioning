import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

from image_captioning.dataset import Sharder, custom_dataloader
from image_captioning.transformer import Decoder
from image_captioning.tokenizer import Tokenizer
from image_captioning.feature_extractor import FeatureExtractor


class CaptionGenerator:
    """
    A class used for generating a caption from file

    Attributes:
        tokenizer (Tokenizer): custom tokenizer
        feature_extractor (FeatureExtractor): pre-trained feature extractor
        decoder (Decoder): decoder used for caption generation
        device (str): string indicating which device will be used for calculations
    """

    def __init__(
        self, decoder: Decoder, tokenizer: Tokenizer, feature_extractor: FeatureExtractor, device: str
    ) -> None:
        """
        Initializes caption generator

        Args:
            tokenizer (Tokenizer): custom tokenizer
            feature_extractor (FeatureExtractor): pre-trained feature extractor
            decoder (Decoder): decoder used for caption generation
            device (str): string indicating which device will be used for calculations
        """

        self.tokenizer: Tokenizer = tokenizer
        self.feature_extractor: FeatureExtractor = feature_extractor
        self.decoder: Decoder = decoder
        self.device: str = device

    def __preprocess_image(self, img_path: str) -> torch.Tensor:
        """
        Method used to read and prepare an image from file

        Args:
            img_path (str): path to the image

        Returns:
            Tensor of features extracted from the image
        """

        raw_image = self.feature_extractor.get_image_from_file(img_path).unsqueeze(0)
        extracted_features = self.feature_extractor.feed(raw_image).squeeze(0)
        extracted_features.to(self.device)

        return extracted_features

    def generate(self, image_path: str, max_size: int, temperature: int = 0.5) -> str:
        """
        Method used to generate a caption

        Args:
            image_path (str): path to the image to generate the caption for
            max_size (int): maximal size of the caption

        Returns:
            String with the generated caption
        """

        max_size = min(max_size, self.tokenizer.max_length)

        generated_caption = self.tokenizer.encode(self.tokenizer.start_token)
        generated_caption = torch.tensor(generated_caption, device=self.device).unsqueeze(0)

        image = self.__preprocess_image(image_path).unsqueeze(0)

        for _ in range(max_size):
            logits = self.decoder(image, generated_caption)
            logits = logits[:, -1, :]  # Fetching the last token of the generated sequence
            predictions = F.softmax(logits, dim=-1)

            if temperature == 0:
                new_token = torch.argmax(predictions, dim=-1).unsqueeze(0)
            else:
                new_token = torch.multinomial(predictions/temperature, num_samples=1)

            generated_caption = torch.cat([generated_caption, new_token], dim=1)
            
            if new_token[0].tolist() == self.tokenizer.encode(Tokenizer.end_token):
                break

        caption_list = generated_caption[0].tolist()
        return self.tokenizer.decode(caption_list)


if __name__ == "__main__":
    pass
