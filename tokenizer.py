import os
import string
from collections import Counter


class Tokenizer:
    """
    Class used for parsing dataset captions' file and tokenizing

    Attributes:
        raw_text (str): text containing lines of pairs (image_name, caption) formatted like: name.jpg#num \t caption
        images_folder (str): path to the folder containing images
        image_paths (list[str]): list of paths to the image files
        captions (list[str]): list of the captions
        word_list (list): list of ordered set of all word tokens
        counter (counter): keeps track of tokens count
        encode_map (dict): dict used to map tokens to indices
        decode_map (dict): dict used to map indices to tokens
        __standardize (bool): whether to transform the caption (like replacing symbols and converting to lowercase)
        __reduce (bool): whether to remove elements from the vocabulary based on some rule
        max_length (int): the maximum length of the caption (including start and end tokens)
    """

    empty_token = ''
    start_token = '<start>'
    end_token = '<end>'
    unknown_token = '<unknown>'

    base_tokens = [empty_token, start_token, end_token, unknown_token]
    base_encode_map = {token: i for i, token in enumerate(base_tokens)}
    base_decode_map = {i: token for i, token in enumerate(base_tokens)}

    def __init__(
            self,
            raw_text: str,
            images_folder: str,
            standardize: bool = True,
            reduce: bool = False
    ) -> None:
        """
        Initializes tokenizer's attributes

        Args:
            raw_text (str): text containing lines of pairs (image_name, caption) formatted like: name.jpg#num \t caption
            images_folder (str): path to the folder containing images
            standardize (bool): whether to transform the caption (like replacing symbols and converting to lowercase)
            reduce (bool): whether to remove elements from the vocabulary based on some rule
        """

        self.raw_text: str = raw_text
        self.images_folder: str = images_folder
        self.image_paths: list[str] = []
        self.captions: list[str] = []
        self.word_list: list[str] = []
        self.counter: Counter = Counter()
        self.encode_map: dict = Tokenizer.base_encode_map
        self.decode_map: dict = Tokenizer.base_decode_map
        self.__standardize: bool = standardize
        self.__reduce: bool = reduce

        self.__extract_captions()
        self.max_length: int = len(max(self.captions, key=len)) + 2  # Plus 2 for <start> and <end> tokens

        if self.__reduce:
            self.__reduce_vocabulary()

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

        line = line.lower()
        line = line.translate(str.maketrans('', '', string.punctuation))  # Removing punctuation
        return line

    def __pad(self, token_list: list[int]) -> list[int]:
        """
        Pads the input list to the maximum length

        Args:
            token_list (list[int]): input token list to pad.

        Returns:
            Input list padded with empty tokens to the size of max_length
        """

        return token_list + (self.max_length - len(token_list)) * [self.encode_map[Tokenizer.empty_token]]

    def __extract_captions(self) -> None:
        """
        Method used to parse the input text
        """

        if len(self.captions) > 0:
            raise AttributeError("Captions have been already extracted from the raw text.")

        for line in self.raw_text.splitlines():
            raw_caption = line.split('\t', 1)
            if len(raw_caption) < 2:
                raise ValueError("Improper line format")

            self.image_paths.append(os.path.join(self.images_folder, raw_caption[0].split('.')[0] + ".jpg"))

            caption = raw_caption[1]
            if self.standardize:
                caption = self.standardize(caption)

            self.counter.update(caption.split())
            self.captions.append(caption)

        self.word_list = [word for word, count in sorted(self.counter.items(), key=lambda x: x[1], reverse=True)]

    def __reduce_vocabulary(self, vocab_size: int = 4996) -> None:
        """
        Reduces the vocabulary to the given size

        Args:
            vocab_size (int): size of the vocabulary list on output (excluding base tokens)
        """

        self.word_list = self.word_list[:vocab_size]
        self.counter = Counter(dict(self.counter.most_common(vocab_size)))

    def __create_mappings(self) -> None:
        """
        Method used to construct mappings based on the current word set
        """

        base_encode_size = len(self.encode_map)
        base_decode_size = len(self.decode_map)

        self.encode_map = self.encode_map | {token: i + base_encode_size for i, token in enumerate(self.word_list)}
        self.decode_map = self.decode_map | {i + base_decode_size: token for i, token in enumerate(self.word_list)}
        self.word_list = Tokenizer.base_tokens + self.word_list

    def encode(self, line_to_encode: str, pad: bool = True) -> list[int]:
        """
        Method used to encode the given string into the list of token indices.

        Note:
            The input does not have to start with <start> and end with <end>.
            Padding with those tokens should be handled before passing string to this method

        Args:
            line_to_encode (str): string to encode
            pad (bool): should the outcome be padded

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

        if pad:
            output_list = self.__pad(output_list)

        return output_list

    def decode(self, list_to_decode: list[int]) -> str:
        """
        Method returning a string constructed from the tokens of given indices

        Args:
            list_to_decode (list[int]): List of tokens' indices

        Returns:
            String constructed from tokens of the given indices
        """
        return ' '.join([self.decode_map[token] for token in list_to_decode])


if __name__ == '__main__':
    file_path = 'dataset/captions.txt'
    folder = 'dataset/images/'

    with open(file_path, "r") as f:
        raw_file = f.read()

    tokenizer = Tokenizer(raw_file, folder, reduce=True)

    print(tokenizer.encode('<START> I am going to work <END>', pad=False))
    print(tokenizer.decode([1, 10, 20, 4, 28, 2]))
