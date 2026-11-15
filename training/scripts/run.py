"""
Launch train.py runs from Python, e.g. for hyperparameter sweeps.

    python training/scripts/run.py
"""

import subprocess
import sys
from pathlib import Path

from infoquality.hyperparameters import HyperParameters

TRAIN_SCRIPT = Path(__file__).parent / "train.py"


def as_args(**kwargs) -> list[str]:
    """{"batch_size": 64} -> ["--batch-size", "64"] (None values are skipped)"""
    args = []
    for k, v in kwargs.items():
        if v is not None:
            args += [f"--{k.replace('_', '-')}", str(v)]
    return args


def train(data_args: dict | None = None, **hyperparameters) -> int:
    """
    Run train.py in a subprocess.

    ### Args
        - `data_args`: data/run arguments, e.g. {"dataset": "imdb"}
        - `hyperparameters`: HyperParameters fields (validated before running)

    ### Returns
        - the exit code of the training process
    """
    hp = HyperParameters(**hyperparameters)
    cmd = [sys.executable, str(TRAIN_SCRIPT)]
    cmd += as_args(**(data_args or {}))
    cmd += as_args(**hp.model_dump(exclude={"num_classes"}))
    print(" \\\n    ".join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode  # noqa: S603


if __name__ == "__main__":
    for lr in (1e-4, 3e-4):
        train(
            data_args={"dataset": "imdb", "fraction": 0.1},
            batch_size=64,
            best_metric="acc",
            dropout=0.4,
            lr=lr,
            early_stopping_patience=2,
            max_len=60,
            model="distilbert-base-uncased",
            name="imdbsent",
            num_epochs=2,
            num_steps=100,
        )
