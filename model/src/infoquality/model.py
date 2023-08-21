import math
from datetime import datetime
from typing import List

import torch
import torch.nn as nn

from infoquality.hyperparameters import HyperParameters
from infoquality.preprocessor import Preprocessor


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len):
        super().__init__()
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model
        )
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.Tensor) -> torch.Tensor:
        return (
            token_embedding
            + self.pos_encoding[  # type: ignore
                0, : token_embedding.size(1), :  # type: ignore
            ]
        )


class Model(nn.Module):
    def __init__(
        self,
        preprocessor: Preprocessor,
        embeddings: torch.Tensor,
        hyperparameters: HyperParameters,
    ):
        super(Model, self).__init__()
        self.hp = hyperparameters
        self.name = self.hp.name
        self.version = datetime.now().strftime(f"{self.hp.version}.%Y%m%d%H%M%S")
        self.preprocessor: Preprocessor = preprocessor
        self.embeddings: torch.Tensor = embeddings.detach().clone()
        self.embedding_dim: int = self.embeddings.shape[1]
        self.num_classes = self.hp.num_classes
        self.dropout_p: float = self.hp.dropout
        self.pos_encoder = PositionalEncoding(
            dim_model=self.embedding_dim,
            max_len=preprocessor.max_len,
        )
        hidden_dim = self.embedding_dim // 2
        self.embedding = nn.Embedding.from_pretrained(
            self.embeddings,
            freeze=False,
            padding_idx=self.preprocessor.pad_idx,
        )
        self.transformer = nn.Transformer(
            d_model=self.embedding_dim,
            nhead=self.hp.num_heads,
            num_encoder_layers=self.hp.num_layers,
            num_decoder_layers=self.hp.num_layers,
            dim_feedforward=hidden_dim,
            dropout=self.hp.dropout,
            batch_first=True,
        )
        self.fc = nn.Linear(self.embedding_dim, self.hp.num_classes)
        self.dropout = nn.Dropout(self.hp.dropout)

    def as_tensors(self, messages: List[str]) -> List[torch.Tensor]:
        return [
            torch.tensor(indices, dtype=torch.long)
            for indices in self.preprocessor(messages)
        ]

    def forward(self, messages: List[str]) -> torch.Tensor:
        indices = self.as_tensors(messages)
        padded = torch.nn.utils.rnn.pad_sequence(
            indices, batch_first=True, padding_value=self.preprocessor.pad_idx
        )
        embedded = self.embedding(padded)
        if self.dropout_p > 0:
            embedded = self.dropout(embedded)
        masked = self.pos_encoder(embedded)
        output = self.transformer(embedded, masked)
        output = self.fc(output)
        return output.mean(dim=1)