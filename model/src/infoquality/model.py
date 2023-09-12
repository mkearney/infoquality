from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from infoquality.hyperparameters import HyperParameters
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging


class Model(nn.Module):
    """
    Initialize a pretrained torch module model

    ### Args
        - `hyperparameters` HyperParameters used to initialize the model.
        - `label_map` A map from target label to target index.
    """

    def __init__(
        self,
        hyperparameters: HyperParameters,
        label_map: Optional[Dict[str, int]] = None,
    ):
        super(Model, self).__init__()
        # model settings
        self.version = datetime.now().strftime(
            f"{hyperparameters.version}.%Y%m%d%H%M%S"
        )
        if label_map:
            self.label_map = label_map
        else:
            self.label_map: Dict[str, int] = {
                str(i): i for i in range(hyperparameters.num_classes)
            }
        self.hyperparameters = hyperparameters
        self.max_len = hyperparameters.max_len

        # model architecture
        logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(hyperparameters.model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            hyperparameters.model,
            num_labels=hyperparameters.num_classes * 3,
            hidden_dropout_prob=hyperparameters.dropout,
        )
        logging.set_verbosity_warning()
        self.linear = nn.Linear(
            hyperparameters.num_classes * 3, hyperparameters.num_classes
        )

    def preprocess(self, messages: List[str]) -> Dict[str, torch.Tensor]:
        """
        Preprocess messages

        ### Args
            - `messages` A list (batch) of messages to be preprocessed.

        ### Returns
            - A dictionary of tokenization results.
        """
        return self.tokenizer(
            messages,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len,
            add_special_tokens=True,
            padding="max_length",
        )  # type: ignore

    def forward(self, messages: List[str]) -> torch.Tensor:
        """
        Forward pass of neural network

        ### Args
            - `messages` A list (batch) of messages to be processed

        ### Returns
            - Tensor of shape (batch_size, num_classes) containing output logits
        """
        inputs = self.preprocess(messages)
        outputs = self.model(**inputs)  # type: ignore
        return self.linear(outputs.logits)
