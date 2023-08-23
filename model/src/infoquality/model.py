from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from infoquality.hyperparameters import HyperParameters
from infoquality.preprocessor import Preprocessor


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len):
        super().__init__()
        # (max_len, dim_model)
        pos_encoding = torch.zeros(max_len, dim_model)
        # (max_len, 1)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        # (dim_model / 2)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * -9.210340371976184 / dim_model
        )
        # (max_len, dim_model / 2) – columns 0, 2, 4...
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        # (max_len, dim_model / 2) – columns 1, 3, 5...
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        # (1, max_len, dim_model)
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
        label_map: Optional[Dict[str, int]] = None,
    ):
        super(Model, self).__init__()
        self.hp = hyperparameters
        self.name = self.hp.name
        self.version = datetime.now().strftime(f"{self.hp.version}.%Y%m%d%H%M%S")
        if not label_map:
            self.label_map: Dict[str, int] = {
                str(i): i for i in range(self.hp.num_classes)
            }
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
        return output.sum(dim=2)
        # return output.mean(dim=1)
