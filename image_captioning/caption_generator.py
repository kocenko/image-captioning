import torch
import torch.nn.functional as F

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

    def __is_generating_done(self, caption: torch.Tensor, max_size: int):
        """
        Method for checking if the generating is over

        Args:
            caption (Tensor): currently generated caption
            max_size (int): maximal size of the caption

        Returns:
            True if generating is over, False otherwise
        """

        size_limit_approached = caption.shape[1] == max_size - 1
        end_token_generated = caption[:, -1] == self.tokenizer.encode_map[self.tokenizer.end_token]

        return size_limit_approached or end_token_generated

    def generate(self, image_path: str, max_size: int) -> str:
        """
        Method used to generate a caption

        Args:
            image_path (str): path to the image to generate the caption for
            max_size (int): maximal size of the caption (excluding beginning and ending tokens)

        Returns:
            String with the generated caption
        """

        max_size = min(max_size + 2, self.tokenizer.max_length)

        generated_caption = self.tokenizer.encode(self.tokenizer.start_token, pad=False)
        generated_caption = torch.tensor(generated_caption, device=self.device).unsqueeze(0)

        image = self.__preprocess_image(image_path).unsqueeze(0)

        while not self.__is_generating_done(generated_caption, max_size):
            logits = self.decoder(image, generated_caption)
            logits = logits[:, -1, :]  # Fetching the last token of the generated sequence
            predictions = F.softmax(logits, dim=-1)
            new_token = torch.multinomial(predictions, num_samples=1)

            generated_caption = torch.cat([generated_caption, new_token], dim=1)

        caption_list = generated_caption[0].tolist()
        return self.tokenizer.decode(caption_list)


if __name__ == "__main__":
    checkpoint = torch.load(
        "../trained/24-10-2023/checkpoint/2023-10-24 14-46-29_45_of_100.pt", map_location=torch.device("cpu")
    )
    hyperparams = checkpoint["hyperparams"]
    hyperparams["device"] = "cpu"

    file_path = "dataset/captions.txt"
    folder = "dataset/images/"
    sample_image_file = "imgs/rooster.jpg"

    with open(file_path, "r") as f:
        raw_file = f.read()
    fe = FeatureExtractor(model_name="mobilenet", device=hyperparams["device"])
    if hyperparams["net_slice_index"]:
        fe.slice_net(hyperparams["net_slice_index"], overwrite_model=True)
    tk = Tokenizer(raw_file, folder, reduce=True)

    model = Decoder(**hyperparams)
    model.load_state_dict(checkpoint["model_state_dict"])

    gener = CaptionGenerator(model, tk, fe, device=hyperparams["device"])
    print(gener.generate(sample_image_file, 20))
