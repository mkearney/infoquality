from typing import Generator, List, Union

import torch
from pydantic import BaseModel

from infoquality.model import Model


class Prediction(BaseModel):
    label: str
    proba: List[float]


class Predictor:
    def __init__(self, model: Model):
        self.model = model
        self.label_map = {v: k for k, v in self.model.label_map.items()}

    def predict(
        self,
        messages: Union[str, List[str]],
        batch_size: int = 50,
    ) -> List[Prediction]:
        return self.predict_batch(
            [messages] if isinstance(messages, str) else messages,
            batch_size,
        )

    def batch(
        self,
        msgs: List[str],
        bs: int,
    ) -> Generator[List[str], List[str], None]:
        for i in range(0, len(msgs), bs):
            yield msgs[i : i + bs]

    def predict_batch(
        self,
        msgs: List[str],
        bs: int,
    ) -> List[Prediction]:
        batches = self.batch(msgs, bs)
        probas = torch.tensor([row for b in batches for row in self.model(b)])
        probas = torch.softmax(probas, 1)
        labels = [self.label_map[i] for i in probas.argmax(1).tolist()]
        return [
            Prediction(label=label, proba=proba.tolist())
            for label, proba in zip(labels, probas)
        ]
