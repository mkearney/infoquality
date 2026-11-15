import json

import pytest
import torch

from infoquality.model import Model
from infoquality.predictor import Prediction, Predictor, batch
from infoquality.save import ModelSaver
from infoquality.utils import get_logger

MESSAGES = ["this is great", "this is not", "meh", "fine I guess", "wow"]


@pytest.mark.parametrize(
    "n, size, expected",
    [(100, 20, 5), (101, 20, 6), (0, 20, 0), (3, 50, 1)],
)
def test_batch(n, size, expected):
    batches = list(batch(["a"] * n, size))
    assert len(batches) == expected
    assert sum(len(b) for b in batches) == n


def test_batch_invalid_size():
    with pytest.raises(ValueError):
        list(batch(["a"], 0))


@pytest.fixture(scope="module")
def saved(model, tmp_path_factory):
    path = tmp_path_factory.mktemp("models")
    saver = ModelSaver(path, version=model.version, logger=get_logger())
    saver.save(model, metrics={"val_loss": [0.5]})
    return saver.path


def test_saved_files(saved, model):
    files = {p.name for p in saved.iterdir()}
    assert files == {
        "state_dict.pt",
        "hyperparameters.json",
        "model.json",
        "metrics.json",
    }
    metadata = json.loads((saved / "model.json").read_text())
    assert metadata == {"version": model.version, "label_map": model.label_map}


def test_saver_refuses_to_overwrite(saved):
    with pytest.raises(FileExistsError):
        ModelSaver(saved.parent, version=saved.name, logger=get_logger())


def test_load_roundtrip(saved, model):
    loaded = Model.load(saved)
    assert not loaded.training
    assert loaded.label_map == model.label_map
    assert loaded.version == model.version
    assert loaded.hyperparameters == model.hyperparameters
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.forward_raw(MESSAGES), model.forward_raw(MESSAGES)
        )


def test_predictor(saved, model):
    predictor = Predictor(str(saved))
    assert predictor.labels == ["low", "medium", "high"]
    preds = predictor.predict(MESSAGES, batch_size=2)
    assert len(preds) == len(MESSAGES)
    assert all(isinstance(p, Prediction) for p in preds)
    assert all(p.label in predictor.labels for p in preds)
    assert all(sum(p.proba) == pytest.approx(1.0) for p in preds)
    # batching does not change results
    probas = predictor.predict_proba(MESSAGES, batch_size=50)
    torch.testing.assert_close(probas, torch.tensor([p.proba for p in preds]))
    with torch.no_grad():
        expected = torch.softmax(model.forward_raw(MESSAGES), 1)
    torch.testing.assert_close(probas, expected)


def test_predictor_single_and_empty(saved):
    predictor = Predictor(saved)
    assert len(predictor.predict("just one")) == 1
    assert predictor.predict([]) == []
