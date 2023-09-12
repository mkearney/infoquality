import json
from typing import Dict, List

from pydantic import BaseModel
from torchmetrics import Accuracy, F1Score, Precision, Recall


class FitStatistics(BaseModel):
    """
    Fit statistics.

    ### Attributes
        - `acc`: accuracy
        - `f1`: f1 score
        - `pr`: precision
        - `rc`: recall
    """

    acc: float
    f1: float
    pr: float
    rc: float


class Fit:
    """
    Fit class.

    ### Args
        - `num_classes`: number of classes
        - `task`: task type
        - `average`: how to average the metrics, i.e., "micro" or "macro"
    """

    def __init__(
        self,
        num_classes: int,
        task: str = "multiclass",
        average: str = "macro",
    ):
        self.num_classes = num_classes
        self.average = average
        self.task = task
        self.accuracy = Accuracy(
            task=self.task,  # type: ignore
            num_classes=num_classes,
            average=self.average,  # type: ignore
        )
        self.f1_score = F1Score(
            task=self.task,  # type: ignore
            num_classes=num_classes,
            average=self.average,  # type: ignore
        )
        self.precision = Precision(
            task=self.task,  # type: ignore
            num_classes=num_classes,
            average=self.average,  # type: ignore
        )
        self.recall = Recall(
            task=self.task,  # type: ignore
            num_classes=num_classes,
            average=self.average,  # type: ignore
        )

    def __call__(self, output, target) -> FitStatistics:
        return FitStatistics(
            acc=self.accuracy(output, target),
            f1=self.f1_score(output, target),
            pr=self.precision(output, target),
            rc=self.recall(output, target),
        )


class EpochMetrics(BaseModel):
    """
    Metrics for a given epoch.

    ### Attributes
        - `epoch`: epoch number
        - `loss`: training loss
        - `val_loss`: validation loss
        - `val_acc`: validation accuracy
        - `val_f1`: validation F1 score
    """

    epoch: int
    loss: float
    val_loss: float
    val_acc: float
    val_f1: float


class Metrics:
    metrics: List[EpochMetrics] = []

    def collect(self) -> Dict[str, List[float]]:
        return {
            "epoch": self.epoch(),
            "loss": self.loss(),
            "val_loss": self.val_loss(),
            "val_acc": self.val_acc(),
            "val_f1": self.val_f1(),
        }

    def epoch(self) -> List[float]:
        return [float(m.epoch) for m in self.metrics]

    def loss(self) -> List[float]:
        return [m.loss for m in self.metrics]

    def val_loss(self) -> List[float]:
        return [m.val_loss for m in self.metrics]

    def val_acc(self) -> List[float]:
        return [m.val_acc for m in self.metrics]

    def val_f1(self) -> List[float]:
        return [m.val_f1 for m in self.metrics]

    def append(self, epoch: int, metrics: Dict[str, float]) -> None:
        epoch_metrics = EpochMetrics(epoch=epoch, **metrics)
        self.metrics.append(epoch_metrics)

    def save(
        self,
        output_dir: str,
        version: str = "",
    ) -> str:
        version = "" if version == "" else f"-{version}"
        save_as = f"{output_dir}/metrics{version}.json"
        data = self.collect()
        with open(save_as, "w") as f:
            json.dump(data, f, indent=2)
        return save_as
