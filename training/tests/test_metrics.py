import pytest
import torch

from common.metrics import Fit, Metrics


def one_hot_logits(preds: list[int], num_classes: int) -> torch.Tensor:
    return torch.nn.functional.one_hot(torch.tensor(preds), num_classes).float()


def test_fit_accumulates_over_batches():
    # class 0: 3 examples all right; class 1: 1 example wrong
    fit = Fit(num_classes=2)
    fit.update(one_hot_logits([0, 0], 2), torch.tensor([0, 0]))
    fit.update(one_hot_logits([0, 0], 2), torch.tensor([0, 1]))
    stats = fit.compute()
    assert stats.acc == pytest.approx(0.75)
    # macro recall: (1.0 + 0.0) / 2
    assert stats.rc == pytest.approx(0.5)


def test_fit_call_is_per_batch():
    fit = Fit(num_classes=3)
    stats = fit(one_hot_logits([0, 1, 2], 3), torch.tensor([0, 1, 2]))
    assert stats.acc == pytest.approx(1.0)
    assert stats.f1 == pytest.approx(1.0)
    stats = fit(one_hot_logits([1, 1, 1], 3), torch.tensor([0, 1, 2]))
    assert stats.acc == pytest.approx(1 / 3)


def epoch(loss: float) -> dict[str, float]:
    return {
        "loss": loss,
        "val_loss": loss,
        "val_acc": 0.5,
        "val_f1": 0.5,
        "val_pr": 0.5,
        "val_rc": 0.5,
    }


def test_metrics_are_per_instance():
    a, b = Metrics(), Metrics()
    a.append(epoch=0, lr=0.1, metrics=epoch(1.0))
    assert len(a.metrics) == 1
    assert len(b.metrics) == 0


def test_metrics_collect():
    m = Metrics()
    m.append(epoch=0, lr=0.1, metrics=epoch(1.0))
    m.append(epoch=1, lr=0.05, metrics=epoch(0.5))
    collected = m.collect()
    assert collected["epoch"] == [0, 1]
    assert collected["lr"] == [0.1, 0.05]
    assert collected["val_loss"] == [1.0, 0.5]
