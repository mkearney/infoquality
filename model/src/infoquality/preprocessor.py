import string
from collections import defaultdict
from typing import List, Union

import torch
from infoquality.artifacts import bert_token_map, tokenizer


class Preprocessor:
    def __init__(self, max_len: int = 128) -> None:
        self.max_len = max_len
        self.unk_tok = "[UNK]"
        self.unk_idx = bert_token_map[self.unk_tok]
        self.pad_tok = "[PAD]"
        self.pad_idx = bert_token_map[self.pad_tok]
        self.token_map = defaultdict(lambda: self.unk_idx)
        self.token_map.update(bert_token_map)
        self.tokenizer = tokenizer

    def rm_punct(self, message: str) -> str:
        return message.translate(str.maketrans("", "", string.punctuation))

    def one_str(self, message: str) -> List[int]:
        message = self.rm_punct(message)
        return self.tokenizer.encode(
            message,
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
        )

    def one(self, x: Union[str, List[int]]) -> List[int]:
        return self.one_str(x) if isinstance(x, str) else x

    def __call__(
        self, messages: Union[List[List[int]], List[str]]
    ) -> List[torch.Tensor]:
        return [
            torch.tensor(self.one(message), dtype=torch.long) for message in messages
        ]
