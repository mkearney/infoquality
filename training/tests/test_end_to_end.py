import json
import random

import polars as pl
import predict
import pytest
import torch
import train

from infoquality.predictor import Predictor

TINY_MODEL = "hf-internal-testing/tiny-random-DistilBertModel"


def make_data(n: int, seed: int) -> pl.DataFrame:
    rng = random.Random(seed)
    labels = [rng.choice(["high", "low", "medium"]) for _ in range(n)]
    return pl.DataFrame(
        {
            "id": list(range(n)),
            "message": [f"{lab} quality message {i}" for i, lab in enumerate(labels)],
            "quality": labels,
        }
    )


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    for split, n in [("train", 48), ("valid", 16), ("test", 16)]:
        make_data(n, seed=n).write_parquet(tmp / f"{split}.parquet")
    args = train.parse_args(
        [
            "--train-path", str(tmp / "train.parquet"),
            "--valid-path", str(tmp / "valid.parquet"),
            "--test-path", str(tmp / "test.parquet"),
            "--text-col", "message",
            "--label-col", "quality",
            "--output-dir", str(tmp / "models"),
            "--device", "cpu",
            "--model", TINY_MODEL,
            "--name", "tiny",
            "--max-len", "12",
            "--batch-size", "8",
            "--num-epochs", "3",
            "--num-steps", "0",
            "--early-stopping-patience", "5",
        ]
    )  # fmt: skip
    return tmp, train.main(args)


def test_train_saves_model_and_metrics(run):
    tmp, results = run
    assert len(results["epochs"]["val_loss"]) == 3
    assert {"loss", "acc", "f1", "pr", "rc"} <= set(results["test"])
    (path,) = (tmp / "models" / "tiny").glob("*/metrics.json")
    metrics = json.loads(path.read_text())
    assert metrics["best_epoch"] == results["best_epoch"]


def test_saved_model_is_the_best_epoch(run):
    tmp, results = run
    best = results["best_epoch"]
    assert results["epochs"]["val_loss"][best] == min(results["epochs"]["val_loss"])
    predictor = Predictor(results["path"])
    assert predictor.labels == ["high", "low", "medium"]


def test_num_classes_mismatch(run):
    tmp, _ = run
    args = train.parse_args(
        ["--train-path", str(tmp / "train.parquet"), "--text-col", "message",
         "--label-col", "quality", "--num-classes", "2", "--model", TINY_MODEL,
         "--output-dir", str(tmp / "models")]
    )  # fmt: skip
    with pytest.raises(ValueError, match="num-classes"):
        train.main(args)


def test_predict_script(run):
    tmp, results = run
    out_path = tmp / "preds.csv"
    args = predict.parse_args(
        [
            "--model-dir", results["path"],
            "--input", str(tmp / "test.parquet"),
            "--output", str(out_path),
            "--text-col", "message",
            "--keep-cols", "id",
            "--device", "cpu",
            "--proba",
        ]
    )  # fmt: skip
    out = predict.main(args)
    assert out.columns == ["id", "label", "proba_high", "proba_low", "proba_medium"]
    assert out.height == 16
    probas = torch.tensor(out.select(pl.col("^proba_.*$")).to_numpy())
    torch.testing.assert_close(probas.sum(1), torch.ones(16, dtype=probas.dtype))
    assert pl.read_csv(out_path).height == 16
