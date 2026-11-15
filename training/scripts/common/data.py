from pathlib import Path

import polars as pl
import torch

from infoquality.data import InputsDataset
from infoquality.model import Model

DATASETS = ("files", "imdb")


def read_table(path: str | Path) -> pl.DataFrame:
    """Read a .parquet, .csv, .tsv, .json, or .jsonl/.ndjson file."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(path)
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix == ".tsv":
        return pl.read_csv(path, separator="\t")
    if suffix == ".json":
        return pl.read_json(path)
    if suffix in (".jsonl", ".ndjson"):
        return pl.read_ndjson(path)
    raise ValueError(f"unsupported file type: {path}")


def standardize(
    df: pl.DataFrame,
    text_col: str = "text",
    label_col: str = "label",
) -> pl.DataFrame:
    """Select/rename columns to `text` and `label` and drop empty texts."""
    missing = {text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(f"missing column(s) {sorted(missing)}; found {df.columns}")
    return (
        df.select(
            pl.col(text_col).cast(pl.String).alias("text"),
            pl.col(label_col).cast(pl.String).alias("label"),
        )
        .drop_nulls()
        .filter(pl.col("text").str.strip_chars().str.len_chars() > 0)
    )


def split(
    df: pl.DataFrame, fraction: float, seed: int
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Shuffle `df` and split it into (first `fraction`, rest)."""
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)
    n = int(df.height * fraction)
    return df.head(n), df.slice(n)


def load_imdb(seed: int) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """IMDB sentiment: HF train -> train; HF test -> 50/50 valid/test."""
    from datasets import load_dataset

    imdb = load_dataset("imdb")
    names = {0: "neg", 1: "pos"}

    def to_df(part: str) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "text": imdb[part]["text"],  # type: ignore
                "label": [names[i] for i in imdb[part]["label"]],  # type: ignore
            }
        )

    valid, test = split(standardize(to_df("test")), 0.5, seed)
    return standardize(to_df("train")), valid, test


def load_splits(
    dataset: str = "files",
    train_path: str | None = None,
    valid_path: str | None = None,
    test_path: str | None = None,
    text_col: str = "text",
    label_col: str = "label",
    valid_size: float = 0.1,
    seed: int = 42,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame | None]:
    """
    Load (train, valid, test) data frames with `text` and `label` columns.

    For `dataset="files"`, `train_path` is required. Without `valid_path`, a
    `valid_size` share of the training data is held out for validation. Without
    `test_path`, test is None and no test set evaluation is done.
    """
    if dataset == "imdb":
        return load_imdb(seed)
    if dataset != "files":
        raise ValueError(f"unknown dataset {dataset!r}; choose from {DATASETS}")
    if not train_path:
        raise ValueError("--train-path is required when --dataset=files")

    def load(path: str) -> pl.DataFrame:
        return standardize(read_table(path), text_col, label_col)

    train = load(train_path)
    if valid_path:
        valid = load(valid_path)
    else:
        valid, train = split(train, valid_size, seed)
    test = load(test_path) if test_path else None
    return train, valid, test


def subsample(df: pl.DataFrame, fraction: float, seed: int) -> pl.DataFrame:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if fraction == 1:
        return df
    return df.sample(fraction=fraction, shuffle=True, seed=seed)


def make_label_map(train: pl.DataFrame, *others: pl.DataFrame | None) -> dict[str, int]:
    """
    Map sorted unique training labels to 0, 1, ...; raise if other splits
    contain labels that are not in the training data.
    """
    labels = sorted(train["label"].unique().to_list())
    for df in others:
        if df is None:
            continue
        unknown = set(df["label"].unique().to_list()) - set(labels)
        if unknown:
            raise ValueError(f"labels not found in training data: {sorted(unknown)}")
    return {label: i for i, label in enumerate(labels)}


def encode(model: Model, df: pl.DataFrame, label_map: dict[str, int]) -> InputsDataset:
    """Tokenize texts and map labels to indices."""
    inputs = model.preprocess(df["text"].to_list())
    targets = torch.tensor(
        df["label"].replace_strict(label_map, return_dtype=pl.Int64).to_list(),
        dtype=torch.int64,
    )
    return InputsDataset(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        targets=targets,
    )
