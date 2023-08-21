from collections import defaultdict
from typing import List

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

    def one(self, message: str) -> List[int]:
        return self.tokenizer.encode(message, truncation=True, max_length=self.max_len)

    def __call__(self, messages: List[str]) -> List[List[int]]:
        return [self.one(message) for message in messages]
