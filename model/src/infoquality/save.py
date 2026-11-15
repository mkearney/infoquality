import json
from pathlib import Path
from typing import Any

import structlog

from infoquality.model import Model


class ModelSaver:
    """
    Saves a model (and its training metrics) to `{path}/{version}`.

    ### Args:
        - `path` (str): Directory to save models to.
        - `version` (str): Version of the model (used as the sub-directory).
        - `logger` (structlog.BoundLogger): Logger.
    """

    def __init__(
        self,
        path: str | Path,
        version: str,
        logger: structlog.BoundLogger,
    ):
        self.path = Path(path).joinpath(version)
        # fail early (before training) rather than overwrite an existing model
        self.path.mkdir(parents=True, exist_ok=False)
        self.logger = logger

    def save_model(self, model: Model) -> None:
        model.save(self.path)
        self.logger.info("__save__", model=str(self.path))

    def save_metrics(self, metrics: dict[str, Any]) -> None:
        path = self.path.joinpath("metrics.json")
        path.write_text(json.dumps(metrics, indent=2))
        self.logger.info("__save__", metrics=str(path))

    def save(self, model: Model, metrics: dict[str, Any]) -> None:
        self.save_model(model)
        self.save_metrics(metrics)
