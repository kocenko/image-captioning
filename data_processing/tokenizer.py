import sys
import string
from collections import Counter
from typing import Optional


class Tokenizer:
    """Class used for parsing captions from the captions read from the dataset file.

    Attributes:
        captions (list[str]): list of the captions
        word_list (list): list of ordered set of all word tokens
        counter (counter): keeps track of tokens count
        encode_map (dict): dict used to map tokens to indices
        decode_map (dict): dict used to map indices to tokens
        max_length (int): the maximum length of the caption (in tokens, including [start] and [end] tokens)
    """

    empty_token = ""
    start_token = "[start]"
    end_token = "[end]"
    unknown_token = "[unknown]"

    base_tokens = [empty_token, start_token, end_token, unknown_token]

    def __init__(self, captions: list[str], max_sequence_size: int = 50, vocabulary_size: Optional[int] = 5000) -> None:
        """Initializes tokenizer's attributes

        Args:
            captions (list[str]): list of captions
            vocabulary_size (int): vocabulary size after reduction
        """

        if max_sequence_size < len(max(captions, key=len).split(" ")):
            print(f"Given max_sequence_size might be too small, resulting in truncating the captions", file=sys.stderr)
        assert max_sequence_size >= 2, "max_sequence_size should be at least 2, to fit [start] and [end] tokens"

        self.max_length: int = max_sequence_size
        self.word_list: list[str] = [Tokenizer.empty_token, Tokenizer.unknown_token]
        self.counter: Counter = Counter()
        self.encode_map: dict = dict()
        self.decode_map: dict = dict()
        self.captions: list[str] = self.prepare_captions(captions)

        if vocabulary_size:
            self.reduce_vocabulary(vocabulary_size)

        self.__create_mappings()

    def truncate(self, line: str) -> str:
        words = line.split(" ")
        if len(words) > self.max_length - 2:
            return " ".join(words[: self.max_length - 2])
        return line

    def standardize(self, line: str) -> str:
        """Converts input line to the unified format

        Consists of:
        - removing punctuation,
        - converting to lower letters,
        - truncating based on max_sequence_size,
        - adding [start] and [end] tokens

        Args:
            line (str): string to standardize

        Returns:
            Standardized string
        """

        line = line.translate(str.maketrans("", "", string.punctuation))
        line = line.lower()
        line = line.strip()
        line = self.truncate(line)
        line = f"{Tokenizer.start_token} {line} {Tokenizer.end_token}"
        return line

    def prepare_captions(self, captions: list[str]) -> list[str]:
        """Method used to standardize captions and prepare word list and counter

        Word list is created based on the frequency of words across all captions.
        Because the padding is performed with [empty] tokens we want to ensure its index remains 0.
        In order to achieve that, this token is excluded from the counter.
        """

        output_captions = []
        for raw_caption in captions:
            caption = self.standardize(raw_caption)
            self.counter.update([word for word in caption.split() if word not in self.word_list])
            output_captions.append(caption)

        self.word_list.extend([word for word, count in sorted(self.counter.items(), key=lambda x: x[1], reverse=True)])
        return output_captions

    def reduce_vocabulary(self, vocab_size: int) -> None:
        """Reduces the vocabulary to the given size

        Args:
            vocab_size (int): size of the vocabulary list on output
        """

        assert vocab_size >= 2, "Vocabulary size after reduction cannot be less than 2 because of base tokens"
        self.word_list = self.word_list[:vocab_size]
        self.counter = Counter(dict(self.counter.most_common(vocab_size - 2)))  # Leaving place for empty and unknown

    def __create_mappings(self) -> None:
        self.encode_map = {token: i for i, token in enumerate(self.word_list)}
        self.decode_map = {i: token for i, token in enumerate(self.word_list)}

    def encode(self, line_to_encode: str, standardize: bool = True, pad: bool = True) -> list[int]:
        """Method used to encode the given string into the list of token indices.

        The input should not start with [start] and end with [end].

        Args:
            line_to_encode (str): string to encode
            standardize (bool): whether to standardize the input first
            pad (bool): whether to pad the output to the max_length

        Returns:
            A list of tokens' indices corresponding to the given string input.
        """

        output_list = []
        word_list = line_to_encode.split() if not standardize else self.standardize(line_to_encode).split()
        output_list = [self.encode_map.get(word, self.encode_map[Tokenizer.unknown_token]) for word in word_list]
        if pad:
            output_list.extend([self.encode_map[Tokenizer.empty_token]] * (self.max_length - len(output_list)))
        return output_list

    def decode(self, list_to_decode: list[int]) -> str:
        """Method returning a string constructed from the tokens of given indices

        Args:
            list_to_decode (list[int]): List of tokens' indices

        Returns:
            String constructed from tokens of the given indices
        """

        return " ".join([self.decode_map[token] for token in list_to_decode])


if __name__ == "__main__":
    pass
