from pydantic import BaseModel


class HyperParameters(BaseModel):
    batch_size: int = 64
    best_metric: str = "loss"
    clip_value: float = 0.0
    dropout: float = 0.2
    early_stopping_patience: int = 4
    gamma: float = 0.67
    lr_patience: int = 0
    lr: float = 2e-05
    max_len: int = 40
    model: str = "distilbert-base-uncased"
    name: str = "nlpmodel"
    num_classes: int = 2
    num_epochs: int = 32
    num_steps: int = 8
    version: str = "0.1.0"
