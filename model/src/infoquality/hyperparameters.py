from enum import Enum

from pydantic import BaseModel


class ActivationEnum(str, Enum):
    relu = "relu"
    gelu = "gelu"


class BestMetric(str, Enum):
    loss = "loss"
    acc = "acc"
    f1 = "f1"


class IndivisibleException(Exception):
    def __init__(self, message: str) -> None:
        self.message = message

    def __str__(self):
        return self.message


class HyperParameters(BaseModel):
    num_epochs: int = 64
    num_steps: int = 64
    batch_size: int = 32
    # dropout: float = 0.1
    lr: float = 1e-4
    gamma: float = 0.67
    max_len: int = 80
    # embedding_dimensions: int = 128
    # num_layers: int = 2
    # num_heads: int = 8
    num_classes: int = 2
    version: str = "0.1.0"
    name: str = "nlpmodel"
    # activation: str = ActivationEnum.relu
    clip_value: float = 9.0
    early_stopping_patience: int = 5
    lr_patience: int = 2
    best_metric: str = "loss"

    # @model_validator(mode="after")
    # def dimensions_must_be_divisible_by_heads(self) -> "HyperParameters":
    #     if not self.embedding_dimensions % self.num_heads == 0:
    #         self.embedding_dimensions = (
    #             self.embedding_dimensions // self.num_heads * self.num_heads
    #         )
    #         try:
    #             raise IndivisibleException(
    #                 f"embedding_dimensions coerced to {self.embedding_dimensions}"
    #                 " to be divisible by num_heads"
    #             )
    #         except IndivisibleException as err:
    #             print(err)
    #     return self
