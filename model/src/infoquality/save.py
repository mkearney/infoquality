import json
from pathlib import Path
from typing import Any, Dict

import torch

from infoquality.model import Model


class ModelSaver:
    def __init__(
        self,
        path: str,
        version: str,
    ):
        self.path = Path(path).joinpath(version)
        self.path.mkdir()

    def save_embeddings(self, model: Model) -> str:
        torch.save(
            model.model.embedding.weight,  # type: ignore
            path := str(self.path.joinpath("embeddings.pt")),
        )
        return path

    def save_state_dict(self, model: Model) -> str:
        torch.save(
            model.state_dict(),
            path := str(self.path.joinpath("state_dict.pt")),
        )
        return path

    def save_hyperparameters(self, model: Model) -> str:
        with open(path := str(self.path.joinpath("hyperparameters.json")), "w") as f:
            json.dump(model.hyperparameters.__dict__, f)
        return path

    def save_metrics(self, metrics: Dict[str, Any]) -> str:
        with open(path := str(self.path.joinpath("metrics.json")), "w") as f:
            json.dump(metrics, f)
        return path
