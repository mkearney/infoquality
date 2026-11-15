from argparse import Namespace

import pytest
import torch

from common.training import EarlyStopping, select_device
from common.utils import format_duration, get_hyperparameters_from_args


def metrics(val_loss: float, val_acc: float) -> dict[str, float]:
    return {"val_loss": val_loss, "val_acc": val_acc}


def test_early_stopping_keeps_a_copy_of_the_best_weights():
    model = torch.nn.Linear(2, 2)
    es = EarlyStopping("loss", patience=2)
    assert es.step(0, metrics(1.0, 0.5), model)
    best = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        model.weight.add_(1.0)
    assert not es.step(1, metrics(2.0, 0.9), model)
    assert es.best_epoch == 0
    torch.testing.assert_close(es.best_state_dict["weight"], best["weight"])


def test_early_stopping_loss_ignores_accuracy():
    # with best_metric="loss", a higher accuracy is not an improvement
    es = EarlyStopping("loss", patience=2)
    model = torch.nn.Linear(1, 1)
    es.step(0, metrics(0.5, 0.6), model)
    assert not es.step(1, metrics(0.6, 0.99), model)
    assert es.best_value == 0.5
    assert es.best_alt_value == 0.6
    assert not es.should_stop
    assert not es.step(2, metrics(0.7, 0.99), model)
    assert es.should_stop


def test_early_stopping_acc():
    es = EarlyStopping("acc", patience=1)
    model = torch.nn.Linear(1, 1)
    es.step(0, metrics(0.5, 0.6), model)
    assert es.step(1, metrics(0.9, 0.7), model)
    assert es.best_epoch == 1
    assert not es.step(2, metrics(0.1, 0.7), model)
    assert es.should_stop


def test_early_stopping_invalid_metric():
    with pytest.raises(ValueError):
        EarlyStopping("f1")


def test_select_device():
    assert select_device("cpu") == torch.device("cpu")
    assert isinstance(select_device("auto"), torch.device)


def test_hyperparameters_from_args():
    args = Namespace(lr=0.1, batch_size=None, fraction=0.5, name="x")
    hp = get_hyperparameters_from_args(args)
    assert hp.lr == 0.1
    assert hp.batch_size == 64  # default
    assert hp.name == "x"


def test_format_duration():
    assert format_duration(30) == {"seconds": "30.00"}
    assert format_duration(600) == {"minutes": "10.00"}
