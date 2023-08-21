import json
import os
from typing import Any, Dict

import torch

from infoquality.model import Model


class ModelSaver:
    def __init__(
        self,
        path: str,
        version: str,
    ):
        self.path = f"{path}/{version}"
        os.mkdir(self.path)

    def save_embeddings(self, model: Model) -> str:
        torch.save(
            model.embedding.weight,
            path := os.path.join(self.path, "embeddings.pt"),
        )
        return path

    def save_state_dict(self, model: Model) -> str:
        torch.save(model.state_dict(), path := os.path.join(self.path, "state_dict.pt"))
        return path

    def save_hyperparameters(self, model: Model) -> str:
        with open(path := os.path.join(self.path, "hyperparameters.json"), "w") as f:
            json.dump(model.hp.__dict__, f)
        return path

    def save_metrics(self, metrics: Dict[str, Any]) -> str:
        with open(path := os.path.join(self.path, "metrics.json"), "w") as f:
            json.dump(metrics, f)
        return path
