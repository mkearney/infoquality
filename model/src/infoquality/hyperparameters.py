from pydantic import BaseModel


class HyperParameters(BaseModel):
    """
    Hyperparameters for language model

    ### Attributes
        - `batch_size`: batch size for training
        - `best_metric`: metric to use for early stopping
        - `clip_value`: gradient clipping value - 0.0 means no clipping
        - `dropout`: dropout rate
        - `early_stopping_patience`: number of consecutive epochs without
            improving best mark
        - `gamma`: discount factor
        - `lr_patience`: patience for learning rate scheduler
        - `lr`: learning rate
        - `max_len`: maximum length of a sequence
        - `model`: model to use
        - `name`: model name
        - `num_classes`: number of classes
        - `num_dims`: number of dimensions
        - `num_epochs`: number of epochs
        - `num_layers`: number of layers
        - `num_steps`: number of steps
        - `version`: model version
    """

    batch_size: int = 64
    best_metric: str = "loss"
    clip_value: float = 0.0
    dropout: float = 0.2
    early_stopping_patience: int = 4
    gamma: float = 0.8
    lr_patience: int = 0
    lr: float = 2e-04
    max_len: int = 40
    model: str = "distilbert-base-uncased"
    name: str = "nlpmodel"
    num_classes: int = 2
    num_dims: int = 0
    num_epochs: int = 32
    num_layers: int = 0
    num_steps: int = 8
    version: str = "0.1.0"
