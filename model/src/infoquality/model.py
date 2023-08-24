import math
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from infoquality.hyperparameters import HyperParameters
from infoquality.preprocessor import Preprocessor


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dimensions, max_len):
        super().__init__()
        # (max_len, embedding_dimensions)
        pos_encoding = torch.zeros(max_len, embedding_dimensions)
        # (max_len, 1)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        # (embedding_dimensions / 2)
        division_term = torch.exp(
            torch.arange(0, embedding_dimensions, 2).float()
            * (-math.log(10000.0))
            / embedding_dimensions
        )
        # (max_len, embedding_dimensions / 2) – columns 0, 2, 4...
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        # (max_len, embedding_dimensions / 2) – columns 1, 3, 5...
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        # (1, max_len, embedding_dimensions)
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

        self.fc = nn.Linear(self.embedding_dimensions, self.hyperparameters.num_classes)
        self.dropout = nn.Dropout(self.hyperparameters.dropout)
        self.layer_norm = nn.LayerNorm(self.embedding_dimensions)

        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.parameters(), lr=self.hyperparameters.lr)
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=self.hyperparameters.gamma
        )

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
        if self.hyperparameters.dropout > 0:
            embedded = self.dropout(embedded)
        embedded = self.layer_norm(embedded)
        masked = self.pos_encoder(embedded)
        transformed = self.transformer_encoder(masked)
        output = self.fc(transformed)
        mean_pooled = F.adaptive_avg_pool1d(output.permute(0, 2, 1), 1).view(
            output.size(0), -1
        )
        max_pooled = F.adaptive_max_pool1d(output.permute(0, 2, 1), 1).view(
            output.size(0), -1
        )
        combined_pooled = mean_pooled + max_pooled
        return combined_pooled

    def train_step(self, messages: List[str], targets: List[int]):
        self.train()
        self.optimizer.zero_grad()
        outputs = self.forward(messages)
        loss = self.loss_fn(outputs, torch.tensor(targets))
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        return loss.item()

    def evaluate(self, messages: List[str], targets: List[int]):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(messages)
            loss = self.loss_fn(outputs, torch.tensor(targets))
            predicted_classes = outputs.argmax(dim=1).tolist()
        return loss.item(), predicted_classes
