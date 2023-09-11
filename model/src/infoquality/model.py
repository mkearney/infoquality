from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from infoquality.hyperparameters import HyperParameters
from transformers import (
    AlbertForSequenceClassification,
    AutoTokenizer,
    BertForSequenceClassification,
    DebertaForSequenceClassification,
    DistilBertForSequenceClassification,
    RobertaForSequenceClassification,
    SqueezeBertForSequenceClassification,
    logging,
)


class Model(nn.Module):
    def __init__(
        self,
        hyperparameters: HyperParameters,
        label_map: Optional[Dict[str, int]] = None,
    ):
        super(Model, self).__init__()
        self.version = datetime.now().strftime(
            f"{hyperparameters.version}.%Y%m%d%H%M%S"
        )
        if label_map:
            self.label_map = label_map
        else:
            self.label_map: Dict[str, int] = {
                str(i): i for i in range(hyperparameters.num_classes)
            }

        logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(hyperparameters.model)
        if hyperparameters.model.startswith("albert-base-v2"):
            self.model = AlbertForSequenceClassification.from_pretrained(
                hyperparameters.model,
                num_labels=hyperparameters.num_classes,  # type: ignore
                use_safetensors=True,  # type: ignore
                # hidden_dropout_prob=hyperparameters.dropout,  # type: ignore
            )
        elif hyperparameters.model.startswith("bert-"):
            self.model = BertForSequenceClassification.from_pretrained(
                hyperparameters.model,
                num_labels=hyperparameters.num_classes,  # type: ignore
                use_safetensors=True,  # type: ignore
                classifier_dropout=hyperparameters.dropout,  # type: ignore
            )
        elif hyperparameters.model.startswith(
            "distilbert-"
        ) or hyperparameters.model.startswith("distilgpt"):
            self.model = DistilBertForSequenceClassification.from_pretrained(
                hyperparameters.model,
                num_labels=hyperparameters.num_classes,  # type: ignore
                use_safetensors=True,  # type: ignore
                seq_classif_dropout=hyperparameters.dropout,  # type: ignore
            )
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            self.model.resize_token_embeddings(len(self.tokenizer))  # type: ignore
        elif hyperparameters.model.startswith(
            "roberta-"
        ) or hyperparameters.model.startswith("distilroberta-"):
            self.model = RobertaForSequenceClassification.from_pretrained(
                hyperparameters.model,
                num_labels=hyperparameters.num_classes,  # type: ignore
                use_safetensors=True,  # type: ignore
                hidden_dropout_prob=hyperparameters.dropout,  # type: ignore
                ignore_mismatched_sizes=True,
            )
        elif hyperparameters.model.startswith("squeezebert"):
            self.model = SqueezeBertForSequenceClassification.from_pretrained(
                hyperparameters.model,
                num_labels=hyperparameters.num_classes,  # type: ignore
                # use_safetensors=True,  # type: ignore
                hidden_dropout_prob=hyperparameters.dropout,  # type: ignore
            )
        elif hyperparameters.model.startswith("microsoft/deberta-base"):
            self.model = DebertaForSequenceClassification.from_pretrained(
                hyperparameters.model,
                num_labels=hyperparameters.num_classes,  # type: ignore
                # use_safetensors=True,  # type: ignore
                hidden_dropout_prob=hyperparameters.dropout,  # type: ignore
            )

        self.hp = hyperparameters
        self.max_len = hyperparameters.max_len
        self.version = datetime.now().strftime("0.0.1-%Y%m%d%H%M%S")
        logging.set_verbosity_warning()
        self.linear = nn.Linear(
            hyperparameters.num_classes + 1, hyperparameters.num_classes
        )

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
