from tokenizer import Tokenizer
from feature_extractor import FeatureExtractor

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader


class ImageCaptionDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, extractor: FeatureExtractor, device: str = "cuda"):
        self.tokenizer: Tokenizer = tokenizer
        self.extractor: FeatureExtractor = extractor
        self.device: str = device

    def __len__(self) -> int:
        return len(self.tokenizer.captions)

    def __getitem__(self, item: int):
        caption = self.tokenizer.encode(f"{self.tokenizer.START_TOKEN} {self.tokenizer.captions[item]} {self.tokenizer.END_TOKEN}")
        input_caption = caption[..., :-1]
        label_caption = caption[..., 1:]
        raw_image = self.extractor.get_image_from_file(self.tokenizer.image_paths[item])
        transformed_image = self.extractor.feed(raw_image.unsqueeze(0).to(self.device)).squeeze(0)
        return transformed_image, input_caption, label_caption


if __name__ == "__main__":
    file_path = 'dataset/captions.txt'
    folder = 'dataset/images/'

    with open(file_path, "r") as f:
        raw_file = f.read()

    fe = FeatureExtractor(device="cpu")
    fe.slice_net(97)
    tk = Tokenizer(raw_file, folder, device="cpu")
    ds = ImageCaptionDataset(tk, fe, device="cpu")
    dl = DataLoader(ds, batch_size=50, shuffle=True)
