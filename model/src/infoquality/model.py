from datetime import datetime
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from infoquality.hyperparameters import HyperParameters
from transformers import AutoModelForSequenceClassification, AutoTokenizer, logging


def change_dropout(config: Dict[str, Any], dropout: float = 0.0) -> Dict[str, Any]:
    """
    Adjust config dictionary with desired dropout

    ### Args:
        - `config` (Dict[str, Any]): config dictionary
        - `dropout` (float, optional): desired dropout. Defaults to 0.0.
    """
    if "seq_classif_dropout" in config:
        config["seq_classif_dropout"] = dropout
    elif "hidden_dropout_prob" in config:
        config["hidden_dropout_prob"] = dropout
    elif "dropout" in config:
        config["dropout"] = dropout
    return config


def change_config(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Adjust config dictionary with named arguments

    ### Args:
        - `config` (Dict[str, Any]): config dictionary
        - `kwargs` (Dict[str, Any]): named arguments
    """
    for param, value in kwargs.items():
        for k, v in config.items():
            if param in k:
                config[k] = value
    return config


def new_model(config_dict: Dict[str, Any]) -> nn.Module:
    """
    Initialize new automodel from config dictionary

    ### Args:
        - `config` (Dict[str, Any]): config dictionary
    """
    return AutoModelForSequenceClassification.from_pretrained(
        config_dict["_name_or_path"],
        **config_dict,
        ignore_mismatched_sizes=True,
    )


def load_model(
    hyperparameters: HyperParameters, label_map: Dict[str, int]
) -> nn.Module:
    """
    Load, adjust, and reload automodel

    ### Args:
        - `hyperparameters` (HyperParameters): desired hyperparameters
        - `label_map` (Dict[str, int]): A map from target label to target index
    """
    rev_label_map = {v: k for k, v in label_map.items()}
    logging.set_verbosity_error()
    init_model = AutoModelForSequenceClassification.from_pretrained(
        hyperparameters.model,
        num_labels=hyperparameters.num_classes,
        id2label=rev_label_map,
        label2id=label_map,
        max_length=hyperparameters.max_len,
    )
    new_args = {
        "max_len": hyperparameters.max_len,
    }
    if hyperparameters.num_layers > 0:
        new_args["n_layers"] = hyperparameters.num_layers
    if hyperparameters.num_dims > 0:
        new_args["dim"] = hyperparameters.num_dims
        new_args["hidden_dim"] = hyperparameters.num_dims * 4
    config_dict = change_config(init_model.config.__dict__, **new_args)
    config_dict = change_dropout(config_dict, dropout=hyperparameters.dropout)
    model = new_model(config_dict)
    logging.set_verbosity_warning()
    return model


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
        self.model = load_model(hyperparameters, label_map=self.label_map)
        logging.set_verbosity_warning()

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

    def forward_raw(self, messages: List[str]) -> torch.Tensor:
        """
        Forward pass of neural network

        ### Args
            - `messages` A list (batch) of messages to be processed

        ### Returns
            - Tensor of shape (batch_size, num_classes) containing output logits
        """
        inputs = self.preprocess(messages)
        return self.forward(**inputs)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of neural network

        ### Args
            - `input_ids` Indices
            - `attention_mask` Masks

        ### Returns
            - Tensor of shape (batch_size, num_classes) containing output logits
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
