import math
from datetime import datetime
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from infoquality.hyperparameters import HyperParameters
from infoquality.preprocessor import Preprocessor


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


class Model(nn.Module):
    def __init__(
        self,
        preprocessor: Preprocessor,
        embeddings: torch.Tensor,
        hyperparameters: HyperParameters,
        label_map: Optional[Dict[str, int]] = None,
    ):
        super(Model, self).__init__()
        self.hyperparameters = hyperparameters
        self.name = self.hyperparameters.name
        self.version = datetime.now().strftime(
            f"{self.hyperparameters.version}.%Y%m%d%H%M%S"
        )
        if not label_map:
            self.label_map: Dict[str, int] = {
                str(i): i for i in range(self.hyperparameters.num_classes)
            }
        self.preprocessor: Preprocessor = preprocessor
        self.embeddings: torch.Tensor = embeddings.detach().clone()
        self.embedding_dimensions: int = self.embeddings.shape[1]
        self.pos_encoder = PositionalEncoding(
            embedding_dimensions=self.embedding_dimensions,
            max_len=preprocessor.max_len,
        )
        hidden_dim = self.embedding_dimensions // 2
        self.embedding = nn.Embedding.from_pretrained(
            self.embeddings,
            freeze=False,
            padding_idx=self.preprocessor.pad_idx,
        )
        self.dropout = nn.Dropout(self.hyperparameters.dropout)
        self.layer_norm = nn.LayerNorm(self.embedding_dimensions)
        self.pos_encoder = PositionalEncoding(
            embedding_dimensions=self.embedding_dimensions,
            max_len=preprocessor.max_len,
        )
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=self.embedding_dimensions,
            nhead=self.hyperparameters.num_heads,
            dim_feedforward=hidden_dim,
            dropout=self.hyperparameters.dropout,
            batch_first=True,
            activation=self.hyperparameters.activation,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=self.hyperparameters.num_layers
        )
        self.fc0 = nn.Linear(
            self.embedding_dimensions, self.hyperparameters.num_classes
        )
        self.fc1 = nn.Linear(
            self.hyperparameters.num_classes * 3, self.hyperparameters.num_classes
        )

    def as_tensors(
        self, messages: Union[List[List[int]], List[str]]
    ) -> List[torch.Tensor]:
        return [
            torch.tensor(indices, dtype=torch.long)
            for indices in self.preprocessor(messages)
        ]

    def forward(self, messages: List[torch.Tensor]) -> torch.Tensor:
        # indices = self.preprocessor(messages)
        padded = torch.nn.utils.rnn.pad_sequence(
            messages, batch_first=True, padding_value=self.preprocessor.pad_idx
        )
        embedded = self.embedding(padded)
        if self.hyperparameters.dropout > 0:
            embedded = self.dropout(embedded)
        embedded = self.layer_norm(embedded)
        masked = self.pos_encoder(embedded)
        transformed = self.transformer_encoder(masked)
        output = self.fc0(transformed)
        mean_pooled = F.adaptive_avg_pool1d(output.permute(0, 2, 1), 1).view(
            output.size(0), -1
        )
        max_pooled = F.adaptive_max_pool1d(output.permute(0, 2, 1), 1).view(
            output.size(0), -1
        )
        std_pooled = output.std(dim=1)
        pooled = torch.cat([mean_pooled, max_pooled, std_pooled], dim=1)
        return self.fc1(pooled)
