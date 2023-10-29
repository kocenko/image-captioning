import string
from collections import Counter
from typing import Optional


class Tokenizer:
    """
    Class used for parsing dataset captions' file and tokenizing

    Attributes:
        captions (list[str]): list of the captions
        word_list (list): list of ordered set of all word tokens
        counter (counter): keeps track of tokens count
        encode_map (dict): dict used to map tokens to indices
        decode_map (dict): dict used to map indices to tokens
        max_length (int): the maximum length of the caption (including start and end tokens)
    """

    empty_token = ""
    start_token = "<start>"
    end_token = "<end>"
    unknown_token = "<unknown>"

    base_tokens = [empty_token, start_token, end_token, unknown_token]

    def __init__(self, captions: list[str], vocabulary_size: Optional[int] = 5000) -> None:
        """
        Initializes tokenizer's attributes

        Args:
            captions (list[str]): list of captions
            vocabulary_size (int): vocabulary size after reduction
        """

        self.word_list: list[str] = [Tokenizer.empty_token, Tokenizer.unknown_token]  # Because it is not read from data
        self.counter: Counter = Counter()
        self.encode_map: dict = dict()
        self.decode_map: dict = dict()
        self.captions: list[str] = self.prepare_captions(captions)
        self.max_length: int = len(max(self.captions, key=len))

        if vocabulary_size:
            self.reduce_vocabulary(vocabulary_size)

        self.__create_mappings()

    @staticmethod
    def standardize(line: str) -> str:
        """
        Converts input line to the unified format

        Args:
            line (str): string to standardize

        Returns:
            Standardized string
        """
        line = line.translate(str.maketrans("", "", string.punctuation))
        line = line.lower()
        line = f"{Tokenizer.start_token} {line} {Tokenizer.end_token}"
        return line

    def prepare_captions(self, captions: list[str]) -> list[str]:
        """
        Method used to standardize captions and prepare word list and counter
        """
        output_captions = []
        for raw_caption in captions:
            caption = self.standardize(raw_caption)
            self.counter.update(caption.split())
            output_captions.append(caption)

        self.word_list.extend([word for word, count in sorted(self.counter.items(), key=lambda x: x[1], reverse=True)])
        return output_captions

    def reduce_vocabulary(self, vocab_size: int) -> None:
        """
        Reduces the vocabulary to the given size

        Args:
            vocab_size (int): size of the vocabulary list on output
        """
        assert vocab_size >= 2, "Vocabulary size after reduction cannot be less than 2 because of base tokens"
        self.word_list = self.word_list[:vocab_size]
        self.counter = Counter(dict(self.counter.most_common(vocab_size - 2)))  # Leaving place for empty and unknown

    def __create_mappings(self) -> None:
        """
        Method used to construct mappings based on the current word set
        """
        self.encode_map = {token: i for i, token in enumerate(self.word_list)}
        self.decode_map = {i: token for i, token in enumerate(self.word_list)}

    def encode(self, line_to_encode: str) -> list[int]:
        """
        Method used to encode the given string into the list of token indices.

        Note:
            The input does not have to start with <start> and end with <end>.
            Padding these tokens should be handled after this method

        Args:
            line_to_encode (str): string to encode

        Returns:
            A list of tokens' indices corresponding to the given string input.
        """

        output_list = []
        word_list = self.standardize(line_to_encode).split()

        for word in word_list:
            if word in self.encode_map:
                output_list.append(self.encode_map[word])
            else:
                output_list.append(self.encode_map[Tokenizer.unknown_token])

        return output_list

    def decode(self, list_to_decode: list[int]) -> str:
        """
        Method returning a string constructed from the tokens of given indices

        Args:
            list_to_decode (list[int]): List of tokens' indices

        Returns:
            String constructed from tokens of the given indices
        """
        return " ".join([self.decode_map[token] for token in list_to_decode])


if __name__ == "__main__":
    pass
