from torch.utils.data import DataLoader

from tokenizer import Tokenizer
from feature_extractor import FeatureExtractor
from dataset import ImageCaptionDataset
from transformer import TransformerBlock


file_path = 'dataset/captions.txt'
folder = 'dataset/images/'
sample_image_file = 'imgs/rooster.jpg'

with open(file_path, "r") as f:
    raw_file = f.read()

# Head size should be equal to embeddings_number // heads_number
model_parameters = {
    "embeddings_number": 64,
    "dropout_rate": 0.2,
    "context_length": 50,
    "blocks_number": 3,
    "heads_number": 4,
    "net_slice_index": 97,
    "device": "cpu"
}

fe = FeatureExtractor(device=model_parameters["device"])
fe.slice_net(model_parameters["net_slice_index"])
tk = Tokenizer(raw_file, folder, device=model_parameters["device"])
ds = ImageCaptionDataset(tk, fe, device=model_parameters["device"])
dl = DataLoader(ds, batch_size=50, shuffle=True)
sample = next(iter(dl))

model_parameters["head_size"] = model_parameters["embeddings_number"] // model_parameters["heads_number"]
model_parameters["vocabulary_size"] = len(tk.word_set)
model_parameters["image_size"] = fe.feed(fe.get_image_from_file(sample_image_file).unsqueeze(0)).shape[2]**2

transformer_block = TransformerBlock(**model_parameters)
output = transformer_block(sample[0], sample[1])
# print(output.shape)

