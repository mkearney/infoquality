from pydantic import BaseModel
from torch import Tensor
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall


class FitStatistics(BaseModel):
    """
    Fit statistics.

    ### Attributes
        - `acc`: accuracy (share of correct predictions)
        - `f1`: macro-averaged F1 score
        - `pr`: macro-averaged precision
        - `rc`: macro-averaged recall (a.k.a. balanced accuracy)
    """

    acc: float
    f1: float
    pr: float
    rc: float


class Fit:
    """
    Accumulates predictions over many batches and computes fit statistics over
    all of them (averaging per-batch macro scores is biased).

    ### Args
        - `num_classes`: number of classes
        - `average`: how to average precision, recall, and F1, i.e., "micro"
            or "macro". Accuracy is always the plain (micro) accuracy.

    ### Example
        fit = Fit(num_classes=2)
        for batch in batches:
            fit.update(model(**batch), batch["targets"])
        stats = fit.compute()
    """

    def __init__(self, num_classes: int, average: str = "macro"):
        self.num_classes = num_classes
        self.average = average
        kwargs = {"task": "multiclass", "num_classes": num_classes}
        self.metrics = MetricCollection(
            {
                "acc": Accuracy(**kwargs, average="micro"),
                "f1": F1Score(**kwargs, average=average),
                "pr": Precision(**kwargs, average=average),
                "rc": Recall(**kwargs, average=average),
            }
        )

    def to(self, device) -> "Fit":
        self.metrics = self.metrics.to(device)
        return self

    def reset(self) -> None:
        self.metrics.reset()

    def update(self, output: Tensor, target: Tensor) -> None:
        self.metrics.update(output, target)

    def compute(self) -> FitStatistics:
        return FitStatistics(**{k: v.item() for k, v in self.metrics.compute().items()})

    def __call__(self, output: Tensor, target: Tensor) -> FitStatistics:
        """Fit statistics for a single batch (does not accumulate)."""
        self.reset()
        self.update(output, target)
        stats = self.compute()
        self.reset()
        return stats


class EpochMetrics(BaseModel):
    """
    Metrics for a given epoch.

    ### Attributes
        - `epoch`: epoch number
        - `lr`: learning rate used during the epoch
        - `loss`: training loss
        - `val_loss`: validation loss
        - `val_acc`: validation accuracy
        - `val_f1`: validation F1 score
        - `val_pr`: validation precision
        - `val_rc`: validation recall
    """

    epoch: int
    lr: float
    loss: float
    val_loss: float
    val_acc: float
    val_f1: float
    val_pr: float
    val_rc: float


class Metrics:
    """Collects `EpochMetrics` over the course of training."""

    def __init__(self) -> None:
        self.metrics: list[EpochMetrics] = []

    def append(self, epoch: int, lr: float, metrics: dict[str, float]) -> None:
        self.metrics.append(EpochMetrics(epoch=epoch, lr=lr, **metrics))

    def collect(self) -> dict[str, list[float]]:
        """Metrics as a dict of columns, e.g. {"val_loss": [0.7, 0.6, ...]}."""
        return {
            field: [getattr(m, field) for m in self.metrics]
            for field in EpochMetrics.model_fields
        }
