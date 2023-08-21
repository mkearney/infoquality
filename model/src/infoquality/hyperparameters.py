from pydantic import BaseModel


class HyperParameters(BaseModel):
    num_epochs: int = 64
    num_steps: int = 8
    batch_size: int = 10
    dropout: float = 0.1
    lr: float = 0.0002
    gamma: float = 0.95
    max_len: int = 128
    embedding_dim: int = 128
    num_layers: int = 2
    num_heads: int = 8
    num_classes: int = 2
    version: str = "0.1.0"
    name: str = "iq"
