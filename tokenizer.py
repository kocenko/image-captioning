import string
from collections import Counter


class Tokenizer:
    START_TOKEN = '<start>'
    END_TOKEN = '<end>'
    UNKNOWN_TOKEN = '<unknown>'

    def __init__(self, raw_text: str, standardize: bool = True, reduce_vocabulary: bool = True):
        self.raw_text: str = raw_text
        self.captions: list[str] = []
        self.word_set: set = set()
        self.word_frequency: Counter = Counter()
        self.encode_map = {Tokenizer.START_TOKEN: 0, Tokenizer.END_TOKEN: 1, Tokenizer.UNKNOWN_TOKEN: 2}
        self.decode_map = {0: Tokenizer.START_TOKEN, 1: Tokenizer.END_TOKEN, 2: Tokenizer.UNKNOWN_TOKEN}
        self.__extract_captions(standardize, reduce_vocabulary)

        if reduce_vocabulary:
            self.__reduce_vocabulary()

        self.__create_mappings()

    @staticmethod
    def __standardize(line: str) -> str:
        line = line.lower()
        line.translate(str.maketrans('', '', string.punctuation))  # Removing punctuation
        return line

    def __extract_captions(self, standardize: bool = True, reduce_vocabulary: bool = True):
        if len(self.captions) > 0:
            raise AttributeError("Captions have been already extracted from the raw text.")

        for line in self.raw_text.splitlines():
            raw_caption = line.split('\t', 1)

            if len(raw_caption) < 2:
                raise ValueError("Improper line format")

            caption = raw_caption[1]

            if standardize:
                caption = self.__standardize(caption)

            self.word_frequency.update(set(caption.split()))
            self.word_set = self.word_set | set(caption.split())
            self.captions.append(caption)

    def __reduce_vocabulary(self):
        pass

    def __create_mappings(self):
        self.encode_map = self.encode_map | {token: i + len(self.encode_map) for i, token in enumerate(self.word_set)}
        self.decode_map = self.decode_map | {i + len(self.decode_map): token for i, token in enumerate(self.word_set)}

    def encode(self, line_to_encode: str) -> list:
        output_list = []
        word_list = [Tokenizer.START_TOKEN] + self.__standardize(line_to_encode).split() + [Tokenizer.END_TOKEN]

        for word in word_list:
            if word in self.encode_map:
                output_list.append(self.encode_map[word])
            else:
                output_list.append(self.encode_map[Tokenizer.UNKNOWN_TOKEN])

        return output_list

    def decode(self, list_to_decode: list) -> str:
        return ' '.join([self.decode_map[token] for token in list_to_decode])


if __name__ == '__main__':
    file_path = 'dataset/captions.txt'

    with open(file_path, "r") as f:
        raw_file = f.read()

    tokenizer = Tokenizer(raw_file)
    # print(tokenizer.encode('I am going to work'))
    # print(tokenizer.decode([0, 10, 20, 4, 28, 1]))
