from collections.abc import Iterator
from pathlib import Path

import torch
from pydantic import BaseModel

from infoquality.model import Model


class Prediction(BaseModel):
    """
    A single prediction.

    ### Attributes
        - `label`: the most probable label
        - `proba`: class probabilities, ordered by label index
    """

    label: str
    proba: list[float]


def batch(messages: list[str], batch_size: int) -> Iterator[list[str]]:
    """Split `messages` into consecutive chunks of at most `batch_size`."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    for i in range(0, len(messages), batch_size):
        yield messages[i : i + batch_size]  # noqa: E203


class Predictor:
    """
    Load a model saved with `Model.save()` and predict labels for messages.

    ### Args
        - `path` Directory the model was saved to
        - `device` Device to run inference on (default: "cpu")
    """

    def __init__(self, path: str | Path, device: str | None = None):
        self.model = load_model(Path(path), device=device or "cpu")
        self.label_map: dict[int, str] = {v: k for k, v in self.model.label_map.items()}

    # kept for backwards compatibility
    batch = staticmethod(batch)

    @property
    def labels(self) -> list[str]:
        """Labels ordered by index (i.e., the order of `Prediction.proba`)."""
        return [self.label_map[i] for i in range(len(self.label_map))]

    def predict_proba(
        self,
        messages: str | list[str],
        batch_size: int = 50,
    ) -> torch.Tensor:
        """Return a (num_messages, num_classes) tensor of probabilities."""
        if isinstance(messages, str):
            messages = [messages]
        num_classes = len(self.label_map)
        if not messages:
            return torch.empty((0, num_classes))
        with torch.inference_mode():
            logits = [self.model.forward_raw(b) for b in batch(messages, batch_size)]
        return torch.softmax(torch.cat(logits), dim=1).cpu()

    def predict(
        self,
        messages: str | list[str],
        batch_size: int = 50,
    ) -> list[Prediction]:
        """Predict a label (and class probabilities) for each message."""
        probas = self.predict_proba(messages, batch_size)
        labels = [self.label_map[i] for i in probas.argmax(1).tolist()]
        return [
            Prediction(label=label, proba=proba)
            for label, proba in zip(labels, probas.tolist(), strict=False)
        ]

    def predict_batch(self, msgs: list[str], bs: int) -> list[Prediction]:
        """Alias of `predict()` kept for backwards compatibility."""
        return self.predict(msgs, batch_size=bs)


def load_model(path: Path, device: str | torch.device = "cpu") -> Model:
    model = Model.load(path, map_location=device)
    return model.to(device)
