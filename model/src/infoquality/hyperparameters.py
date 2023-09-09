from enum import Enum

from pydantic import BaseModel


class BestMetric(str, Enum):
    loss = "loss"
    acc = "acc"
    f1 = "f1"


class HyperParameters(BaseModel):
    num_epochs: int = 64
    num_steps: int = 64
    batch_size: int = 32
    dropout: float = 0.1
    lr: float = 1e-4
    gamma: float = 0.67
    max_len: int = 80
    num_classes: int = 2
    version: str = "0.1.0"
    name: str = "nlpmodel"
    clip_value: float = 9.0
    early_stopping_patience: int = 5
    lr_patience: int = 2
    best_metric: str = "loss"
    version: str = "0.1.0"
