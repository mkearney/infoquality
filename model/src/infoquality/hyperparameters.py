from enum import Enum

from pydantic import BaseModel


class BestMetric(str, Enum):
    acc = "acc"
    f1 = "f1"
    loss = "loss"


class HyperParameters(BaseModel):
    batch_size: int = 32
    best_metric: str = "loss"
    clip_value: float = 9.0
    dropout: float = 0.1
    early_stopping_patience: int = 5
    gamma: float = 0.67
    lr_patience: int = 2
    lr: float = 1e-4
    max_len: int = 80
    name: str = "nlpmodel"
    num_classes: int = 2
    num_epochs: int = 64
    num_steps: int = 64
    version: str = "0.1.0"
