import torch
import torch.nn.functional as F
from transformer import Decoder
from tokenizer import Tokenizer
from feature_extractor import FeatureExtractor


class CaptionGenerator:
    def __init__(self, decoder: Decoder, tokenizer: Tokenizer, feature_extractor: FeatureExtractor, **kwargs):
        self.tokenizer: Tokenizer = tokenizer
        self.feature_extractor: FeatureExtractor = feature_extractor
        self.config = kwargs
        self.decoder: Decoder = decoder

    def __preprocess_image(self, img_path: str) -> torch.Tensor:
        raw_image = self.feature_extractor.get_image_from_file(img_path).unsqueeze(0)
        extracted_features = self.feature_extractor.feed(raw_image).squeeze(0)
        return extracted_features

    def generate(self, image_path: str, max_size: int) -> str:
        image = self.__preprocess_image(image_path)
        start_vector = torch.tensor(self.tokenizer.encode(self.tokenizer.start_token)).unsqueeze(0)
        start_vector = start_vector[:, :1]
        image = image.unsqueeze(0)
        max_size = min(max_size + 2, self.tokenizer.max_length)

        while True:
            if start_vector.shape[1] == max_size - 1:
                new_token = torch.tensor(self.tokenizer.encode(self.tokenizer.end_token))[:1].unsqueeze(0)
            else:
                logits = self.decoder(image, start_vector)
                logits = logits[:, -1, :]
                predictions = F.softmax(logits, dim=-1)
                new_token = torch.multinomial(predictions, num_samples=1)

            start_vector = torch.cat([start_vector, new_token], dim=1)

            if new_token.detach().numpy() == self.tokenizer.encode_map[self.tokenizer.end_token]:
                break

        return self.tokenizer.decode(start_vector[0, 1:-1].detach().numpy().tolist())