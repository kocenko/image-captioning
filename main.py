from tokenizer import Tokenizer
from feature_extractor import FeatureExtractor
from dataset import ImageCaptionDataset
from train import Trainer
from transformer import Decoder

from torch.utils.tensorboard import SummaryWriter

file_path = 'dataset/captions.txt'
folder = 'dataset/images/'
sample_image_file = 'imgs/rooster.jpg'
checkpoint_path = 'checkpoints/'
summary_folder = 'summary/'

with open(file_path, "r") as f:
    raw_file = f.read()

# Head size should be equal to embeddings_number // heads_number
hyperparameters = {
    "batches": 50,
    "split_lengths": (.7, .2, .1),
    "banned_tokens": [0, 2, 3],
    "embeddings_number": 64,
    "dropout_rate": 0.2,
    "learning_rate": 10e-4,
    "epochs": 2,
    "blocks_number": 3,
    "heads_number": 4,
    "head_size": 16,
    "net_slice_index": 97,
    "eval_iterations": 200,
    "eval_per_epoch": 20,
    "device": "cpu"
}

fe = FeatureExtractor(device=hyperparameters["device"])
fe.slice_net(hyperparameters["net_slice_index"])
tk = Tokenizer(raw_file, folder)
ds = ImageCaptionDataset(tk, fe, device=hyperparameters["device"])
wr = SummaryWriter(summary_folder)

# Updating dependent hyperparameters
hyperparameters["vocabulary_size"] = len(tk.word_list)
hyperparameters["context_length"] = tk.max_length
hyperparameters["image_channels"] = fe.feed(fe.get_image_from_file(sample_image_file).unsqueeze(0)).shape[1]
hyperparameters["word_count"] = tk.counter
hyperparameters["encode_map"] = tk.encode_map


dec = Decoder(**hyperparameters)

trainer = Trainer(tk, fe, ds, checkpoint_path, sample_image_file, wr, hyperparameters)
trainer.train()
