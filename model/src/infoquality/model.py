import json
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    logging,
)

from infoquality.hyperparameters import HyperParameters

# Config keys used by different architectures for the same concept. The first
# key present in a model's config is the one that gets overridden.
LAYER_KEYS = ("n_layers", "num_hidden_layers", "n_layer", "num_layers")
DIM_KEYS = ("dim", "hidden_size", "n_embd", "d_model")
HIDDEN_DIM_KEYS = ("hidden_dim", "intermediate_size", "n_inner", "ffn_dim")
CLASSIFIER_DROPOUT_KEYS = (
    "seq_classif_dropout",
    "classifier_dropout",
    "hidden_dropout_prob",
    "dropout",
)

# Files written by `Model.save()` / read by `Model.load()`
STATE_DICT_FILE = "state_dict.pt"
HYPERPARAMETERS_FILE = "hyperparameters.json"
METADATA_FILE = "model.json"


def _first_key(config: PretrainedConfig, keys: tuple) -> str | None:
    return next((k for k in keys if hasattr(config, k)), None)


def build_config(
    hyperparameters: HyperParameters,
    label_map: dict[str, int],
) -> PretrainedConfig:
    """
    Build a transformers config for `hyperparameters.model` with the labels,
    dropout, and (optionally) architecture overrides applied.

    ### Args:
        - `hyperparameters` (HyperParameters): desired hyperparameters
        - `label_map` (Dict[str, int]): A map from target label to target index
    """
    config = AutoConfig.from_pretrained(
        hyperparameters.model,
        num_labels=len(label_map),
        id2label={v: k for k, v in label_map.items()},
        label2id=dict(label_map),
    )
    overrides: dict[str, int | float] = {}
    if hyperparameters.num_layers > 0:
        if key := _first_key(config, LAYER_KEYS):
            overrides[key] = hyperparameters.num_layers
    if hyperparameters.num_dims > 0:
        if key := _first_key(config, DIM_KEYS):
            overrides[key] = hyperparameters.num_dims
        if key := _first_key(config, HIDDEN_DIM_KEYS):
            overrides[key] = hyperparameters.num_dims * 4
    if key := _first_key(config, CLASSIFIER_DROPOUT_KEYS):
        overrides[key] = hyperparameters.dropout
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def build_model(
    hyperparameters: HyperParameters,
    label_map: dict[str, int],
    pretrained: bool = True,
) -> PreTrainedModel:
    """
    Initialize a sequence classification model.

    ### Args
        - `hyperparameters` (HyperParameters): desired hyperparameters
        - `label_map` (Dict[str, int]): A map from target label to target index
        - `pretrained` (bool): load pretrained weights (True) or initialize
            randomly (False, e.g. when weights are loaded from a state dict)
    """
    config = build_config(hyperparameters, label_map)
    if not pretrained:
        return AutoModelForSequenceClassification.from_config(config)
    if hyperparameters.num_dims > 0:
        warnings.warn(
            "num_dims > 0 changes the hidden size, so the pretrained weights of "
            "mismatched layers are discarded and randomly re-initialized.",
            stacklevel=2,
        )
    return AutoModelForSequenceClassification.from_pretrained(
        hyperparameters.model,
        config=config,
        ignore_mismatched_sizes=True,
    )


def default_label_map(num_classes: int) -> dict[str, int]:
    return {str(i): i for i in range(num_classes)}


class Model(nn.Module):
    """
    A pretrained transformer for sequence classification plus its tokenizer.

    ### Args
        - `hyperparameters` HyperParameters used to initialize the model.
        - `label_map` A map from target label to target index. Defaults to
            `{"0": 0, "1": 1, ...}`.
        - `version` Model version. Defaults to
            `{hyperparameters.version}.{timestamp}`.
        - `pretrained` Load pretrained weights (True) or initialize randomly.
    """

    def __init__(
        self,
        hyperparameters: HyperParameters,
        label_map: dict[str, int] | None = None,
        version: str | None = None,
        pretrained: bool = True,
    ):
        super().__init__()
        label_map = label_map or default_label_map(hyperparameters.num_classes)
        if sorted(label_map.values()) != list(range(len(label_map))):
            raise ValueError("label_map values must be 0, 1, ..., n - 1")
        if len(label_map) != hyperparameters.num_classes:
            raise ValueError(
                f"label_map has {len(label_map)} labels but "
                f"num_classes={hyperparameters.num_classes}"
            )
        self.label_map: dict[str, int] = dict(label_map)
        self.hyperparameters = hyperparameters
        self.max_len = hyperparameters.max_len
        self.version = version or datetime.now().strftime(
            f"{hyperparameters.version}.%Y%m%d%H%M%S"
        )

        previous_verbosity = logging.get_verbosity()
        logging.set_verbosity_error()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(hyperparameters.model)
            self.model = build_model(hyperparameters, self.label_map, pretrained)
        finally:
            logging.set_verbosity(previous_verbosity)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def preprocess(self, messages: list[str]) -> dict[str, torch.Tensor]:
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

    def forward_raw(self, messages: list[str]) -> torch.Tensor:
        """
        Tokenize messages and run a forward pass.

        ### Args
            - `messages` A list (batch) of messages to be processed

        ### Returns
            - Tensor of shape (batch_size, num_classes) containing output logits
        """
        inputs = self.preprocess(messages)
        return self.forward(
            input_ids=inputs["input_ids"].to(self.device),
            attention_mask=inputs["attention_mask"].to(self.device),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass of neural network

        ### Args
            - `input_ids` Token indices
            - `attention_mask` Attention masks
            - `targets` Ignored; accepted so dataset batches can be unpacked
                directly with `model(**batch)`.

        ### Returns
            - Tensor of shape (batch_size, num_classes) containing output logits
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

    def save(self, path: str | Path) -> Path:
        """
        Save weights, hyperparameters, version, and label map to directory `path`.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / STATE_DICT_FILE)
        hp = self.hyperparameters.model_dump()
        (path / HYPERPARAMETERS_FILE).write_text(json.dumps(hp, indent=2))
        metadata = {"version": self.version, "label_map": self.label_map}
        (path / METADATA_FILE).write_text(json.dumps(metadata, indent=2))
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str | torch.device | None = "cpu",
    ) -> "Model":
        """
        Load a model saved with `Model.save()` and put it in eval mode.

        ### Args
            - `path` Directory the model was saved to
            - `map_location` Device to load weights onto (default: "cpu")
        """
        path = Path(path)
        hp = HyperParameters(**json.loads((path / HYPERPARAMETERS_FILE).read_text()))
        # models saved by older versions of this package have no metadata file
        metadata = {}
        if (path / METADATA_FILE).exists():
            metadata = json.loads((path / METADATA_FILE).read_text())
        version = metadata.get("version")
        label_map = metadata.get("label_map")
        model = cls(hp, label_map=label_map, version=version, pretrained=False)
        state_dict = torch.load(
            path / STATE_DICT_FILE, map_location=map_location, weights_only=True
        )
        model.load_state_dict(state_dict)
        model.eval()
        return model
