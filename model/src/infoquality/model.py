import json
import math
from datetime import datetime
from typing import Dict, List

import torch
import torch.nn as nn
from infoquality.hyperparameters import HyperParameters
from transformers import AutoTokenizer, DistilBertForSequenceClassification, logging

# x = "This is a test sentence."


class Model(nn.Module):
    def __init__(
        self, hyperparameters: HyperParameters, model: str = "distilbert-base-uncased"
    ):
        super(Model, self).__init__()
        logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model, num_labels=10, use_safetensors=True
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


# class Model(nn.Module):
#     def __init__(
#         self,
#         preprocessor: Preprocessor,
#         hyperparameters: HyperParameters,
#         label_map: Optional[Dict[str, int]] = None,
#     ):
#         super(Model, self).__init__()
#         self.hyperparameters = hyperparameters
#         self.name = self.hyperparameters.name
#         self.version = datetime.now().strftime(
#             f"{self.hyperparameters.version}.%Y%m%d%H%M%S"
#         )
#         if not label_map:
#             self.label_map: Dict[str, int] = {
#                 str(i): i for i in range(self.hyperparameters.num_classes)
#             }
#         self.preprocessor: Preprocessor = preprocessor
#         self.embeddings: torch.Tensor = embeddings.detach().clone()
#         self.embedding_dimensions: int = self.embeddings.shape[1]
#         hidden_dim = self.embedding_dimensions
#         self.embedding = nn.Embedding.from_pretrained(
#             self.embeddings,
#             freeze=False,
#             padding_idx=self.preprocessor.tokenizer.unk_token_id,
#         )
#         self.dropout = nn.Dropout(self.hyperparameters.dropout)
#         self.layer_norm = nn.LayerNorm(self.embedding_dimensions)
#         self.gru = nn.GRU(
#             input_size=self.embedding_dimensions,
#             hidden_size=hidden_dim,
#             num_layers=self.hyperparameters.num_layers,
#             batch_first=True,
#             dropout=self.hyperparameters.dropout,
#             bidirectional=True,
#         )
#         self.fc = nn.Linear(
#             self.embedding_dimensions * 2 * 3, self.hyperparameters.num_classes
#         )

#     def as_tensors(
#         self, messages: Union[List[List[int]], List[str]]
#     ) -> List[torch.Tensor]:
#         return [
#             torch.tensor(indices, dtype=torch.long)
#             for indices in self.preprocessor(messages)
#         ]

#     def forward(self, messages: List[torch.Tensor]) -> torch.Tensor:
#         padded = torch.nn.utils.rnn.pad_sequence(
#             messages,
#             batch_first=True,
#         )
#         embedded = self.embedding(padded)
#         if self.hyperparameters.dropout > 0:
#             embedded = self.dropout(embedded)
#         embedded = self.layer_norm(embedded)
#         # gru
#         output, _ = self.gru(embedded)
#         output = output.permute(0, 2, 1)
#         avg_pooled = F.adaptive_avg_pool1d(output, 1).view(output.size(0), -1)
#         max_pooled = F.adaptive_max_pool1d(output, 1).view(output.size(0), -1)
#         std_pooled = output.std(dim=-1)
#         pooled = torch.cat([avg_pooled, max_pooled, std_pooled], dim=1)
#         return self.fc(pooled)
