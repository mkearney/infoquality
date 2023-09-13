import json
from pathlib import Path
from typing import Any, Dict

import structlog
import torch

from infoquality.model import Model


class ModelSaver:
    """
    Saves pytorch model to disk.

    ### Args:
        - `path` (str): Path to save the model.
        - `version` (str): Version of the model.
    """

    def __init__(self, path: str, version: str, logger: structlog.BoundLogger):
        self.path = Path(path).joinpath(version)
        self.path.mkdir()
        self.logger = logger

    def save_state_dict(self, model: Model) -> None:
        torch.save(
            model.state_dict(),
            path := str(self.path.joinpath("state_dict.pt")),
        )
        self.logger.info("__save__", state_dict=path)

    def save_hyperparameters(self, model: Model) -> None:
        with open(path := str(self.path.joinpath("hyperparameters.json")), "w") as f:
            json.dump(model.hyperparameters.__dict__, f)
        self.logger.info("__save__", hyperparameters=path)

    def save_metrics(self, metrics: Dict[str, Any]) -> None:
        with open(path := str(self.path.joinpath("metrics.json")), "w") as f:
            json.dump(metrics, f)
        self.logger.info("__save__", metrics=path)

    def save(self, model: Model, metrics: Dict[str, Any]) -> None:
        self.save_state_dict(model)
        self.save_hyperparameters(model)
        self.save_metrics(metrics)
