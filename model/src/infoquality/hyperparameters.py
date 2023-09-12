from pydantic import BaseModel


class HyperParameters(BaseModel):
    """
    Hyperparameters for language model

    Attributes:
    ----------
    batch_size: batch size for training
    best_metric: metric to use for early stopping
    clip_value: gradient clipping value - 0.0 means no clipping
    dropout: dropout rate
    early_stopping_patience: number of consecutive epochs without
        improving best mark
    fraction: fraction of training data to use for validation
    gamma: discount factor
    lr_patience: patience for learning rate scheduler
    lr: learning rate
    max_len: maximum length of a sequence
    model: model to use
    name: model name
    num_classes: number of classes
    num_epochs: number of epochs
    num_steps: number of steps
    version: model version
    """

    batch_size: int = 64
    best_metric: str = "loss"
    clip_value: float = 0.0
    dropout: float = 0.2
    early_stopping_patience: int = 4
    fraction: float = 1.0
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
