from tokenizer import Tokenizer
from feature_extractor import FeatureExtractor
from dataset import Sharder
from train import Trainer

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
    "batches": 32,
    "split_lengths": (.7, .2, .1),
    "banned_tokens": [0, 1, 3],
    "embeddings_number": 256,
    "dropout_rate": 0.1,
    "learning_rate": 1e-4,
    "epochs": 20,
    "blocks_number": 1,
    "heads_number": 1,
    "head_size": 256,
    "net_slice_index": "layers.15",
    "eval_iterations": 10,
    "eval_per_epoch": 10,
    "device": "cpu"
}

fe = FeatureExtractor(model_name="mnasnet0_75", device=hyperparameters["device"])
if hyperparameters["net_slice_index"]:
    fe.slice_net(hyperparameters["net_slice_index"], overwrite_model=True)
tk = Tokenizer(raw_file, folder, reduce=True)
sh = Sharder(tk, fe, batch_size=hyperparameters["batches"], shard_size=2000, device=hyperparameters["device"])
# sh.save_shards()
wr = SummaryWriter(summary_folder)

# Updating dependent hyperparameters
hyperparameters["vocabulary_size"] = len(tk.word_list)
hyperparameters["context_length"] = tk.max_length
hyperparameters["image_channels"] = fe.feed(fe.get_image_from_file(sample_image_file).unsqueeze(0)).shape[1]
hyperparameters["word_count"] = tk.counter

trainer = Trainer(tk, fe, sh, checkpoint_path, sample_image_file, wr, hyperparameters)
trainer.train()
