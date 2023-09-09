import json
import math
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from infoquality.hyperparameters import HyperParameters
from transformers import AutoTokenizer, DistilBertForSequenceClassification, logging


class Model(nn.Module):
    def __init__(
        self,
        hyperparameters: HyperParameters,
        model: str = "distilbert-base-uncased",
        label_map: Optional[Dict[str, int]] = None,
    ):
        super(Model, self).__init__()
        self.version = datetime.now().strftime(
            f"{self.hyperparameters.version}.%Y%m%d%H%M%S"
        )
        if label_map:
            self.label_map = label_map
        else:
            self.label_map: Dict[str, int] = {
                str(i): i for i in range(self.hyperparameters.num_classes)
            }

        logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model,
            num_labels=hyperparameters.num_classes,
            use_safetensors=True,
            seq_classif_dropout=hyperparameters.dropout,
        )
        self.hp = hyperparameters
        self.max_len = hyperparameters.max_len
        self.version = datetime.now().strftime("0.0.1-%Y%m%d%H%M%S")
        logging.set_verbosity_warning()

    def preprocess(self, messages: List[str]) -> Dict[str, torch.Tensor]:
        return self.tokenizer(
            messages,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
        )  # type: ignore

    def forward(self, messages: List[str]) -> torch.Tensor:
        inputs = self.preprocess(messages)
        outputs = self.model(**inputs)  # type: ignore
        return outputs.logits


with open("/Users/mwk/data/kmw-dbcr-token-map.json", "r") as f:
    token_map = json.load(f)

embeddings = torch.load("/Users/mwk/data/kmw-dbcr-embeddings.pt")


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dimensions, max_len):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dimensions, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embedding_dimensions)
        )
        pe = torch.zeros(max_len, embedding_dimensions)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pos_encoding", pe)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        return (
            token_embedding
            + self.pos_encoding[  # type: ignore
                0, : token_embedding.size(1), :  # type: ignore
            ]
        )
