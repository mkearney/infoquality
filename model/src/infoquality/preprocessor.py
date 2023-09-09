from typing import List, Union

import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


class Preprocessor:
    def __init__(self, max_len: int = 128) -> None:
        self.max_len = max_len
        self.tokenizer = tokenizer

    def one_str(self, message: str) -> List[int]:
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
