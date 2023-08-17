from torch.utils.data import DataLoader

from tokenizer import Tokenizer
from feature_extractor import FeatureExtractor
from dataset import ImageCaptionDataset
from transformer import EncoderBlock


file_path = 'dataset/captions.txt'
folder = 'dataset/images/'

with open(file_path, "r") as f:
    raw_file = f.read()

fe = FeatureExtractor(device="cpu")
fe.slice_net(97)
tk = Tokenizer(raw_file, folder, device="cpu")
ds = ImageCaptionDataset(tk, fe, device="cpu")
dl = DataLoader(ds, batch_size=50, shuffle=True)

encoder = EncoderBlock()
sample = next(iter(dl))

print(sample[0].shape)
output = encoder(sample[0])
print(output.shape)
