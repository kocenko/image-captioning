import os
import string
from collections import Counter

import torch


class Tokenizer:
    EMPTY_TOKEN = ''
    START_TOKEN = '<start>'
    END_TOKEN = '<end>'
    UNKNOWN_TOKEN = '<unknown>'

    def __init__(
            self, raw_text: str,
            images_folder: str,
            standardize: bool = True,
            reduce_vocabulary: bool = True,
            device: str = "cuda"
    ) -> None:

        self.raw_text: str = raw_text
        self.image_paths: list[str] = []
        self.captions: list[str] = []
        self.word_set: set = set()
        self.word_frequency: Counter = Counter()
        self.max_length: int = 0
        self.device: str = device
        self.images_folder: str = images_folder

        start_token = Tokenizer.START_TOKEN
        end_token = Tokenizer.END_TOKEN
        unknown_token = Tokenizer.UNKNOWN_TOKEN
        empty_token = Tokenizer.EMPTY_TOKEN

        self.encode_map = {start_token: 0, end_token: 1, unknown_token: 2, empty_token: 3}
        self.decode_map = {0: start_token, 1: end_token, 2: unknown_token, 3: empty_token}
        self.__extract_captions(standardize, reduce_vocabulary)

        if reduce_vocabulary:
            self.__reduce_vocabulary()

        self.__create_mappings()

    @staticmethod
    def __standardize(line: str) -> str:
        line = line.lower()
        line.translate(str.maketrans('', '', string.punctuation))  # Removing punctuation
        return line

    def __pad_tensor(self, token_list: list[int]) -> torch.Tensor:
        token_list = token_list + (self.max_length - len(token_list)) * [self.encode_map[Tokenizer.EMPTY_TOKEN]]
        return torch.tensor(token_list, device=self.device)

    def __extract_captions(self, standardize: bool = True, reduce_vocabulary: bool = True):
        if len(self.captions) > 0:
            raise AttributeError("Captions have been already extracted from the raw text.")

        for line in self.raw_text.splitlines():
            raw_caption = line.split('\t', 1)
            if len(raw_caption) < 2:
                raise ValueError("Improper line format")

            self.image_paths.append(os.path.join(self.images_folder, raw_caption[0].split('#')[0]))
            caption = raw_caption[1]
            if standardize:
                caption = self.__standardize(caption)

            if len(caption) > self.max_length:
                self.max_length = len(caption)

            captions_set = set(caption.split())
            if reduce_vocabulary:
                self.word_frequency.update(captions_set)

            self.word_set = self.word_set | captions_set
            self.captions.append(caption)

    def __reduce_vocabulary(self):
        pass

    def __create_mappings(self):
        self.encode_map = self.encode_map | {token: i + len(self.encode_map) for i, token in enumerate(self.word_set)}
        self.decode_map = self.decode_map | {i + len(self.decode_map): token for i, token in enumerate(self.word_set)}

    def encode(self, line_to_encode: str) -> torch.Tensor:
        output_list = []
        word_list = [Tokenizer.START_TOKEN] + self.__standardize(line_to_encode).split() + [Tokenizer.END_TOKEN]

        for word in word_list:
            if word in self.encode_map:
                output_list.append(self.encode_map[word])
            else:
                output_list.append(self.encode_map[Tokenizer.UNKNOWN_TOKEN])

        return self.__pad_tensor(output_list)

    def decode(self, list_to_decode: list) -> str:
        return ' '.join([self.decode_map[token] for token in list_to_decode])


if __name__ == '__main__':
    file_path = 'dataset/captions.txt'
    folder = 'dataset/images/'

    with open(file_path, "r") as f:
        raw_file = f.read()

    tokenizer = Tokenizer(raw_file, folder)
    print(tokenizer.encode('I am going to work'))
    # print(tokenizer.decode([0, 10, 20, 4, 28, 1]))
