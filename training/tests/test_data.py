import polars as pl
import pytest

from common.data import load_splits, make_label_map, read_table, standardize


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "body": ["good text", "bad", "  ", None, "fine", "ok", "meh", "great"],
            "y": ["high", "low", "low", "high", "high", "low", "low", "high"],
        }
    )


def test_standardize_renames_and_drops_empty(df):
    out = standardize(df, text_col="body", label_col="y")
    assert out.columns == ["text", "label"]
    assert out.height == 6


def test_standardize_missing_column(df):
    with pytest.raises(ValueError, match="missing column"):
        standardize(df)


@pytest.mark.parametrize("suffix", [".parquet", ".csv", ".jsonl"])
def test_read_table(df, tmp_path, suffix):
    path = tmp_path / f"data{suffix}"
    {
        ".parquet": df.write_parquet,
        ".csv": df.write_csv,
        ".jsonl": df.write_ndjson,
    }[suffix](path)
    assert read_table(path).height == df.height


def test_load_splits_holds_out_validation(df, tmp_path):
    path = tmp_path / "train.parquet"
    df.write_parquet(path)
    train, valid, test = load_splits(
        train_path=str(path), text_col="body", label_col="y", valid_size=0.5
    )
    assert train.height == valid.height == 3
    assert test is None
    assert set(train["text"]).isdisjoint(valid["text"])


def test_load_splits_requires_train_path():
    with pytest.raises(ValueError, match="train-path"):
        load_splits()


def test_make_label_map_sorted_and_validated():
    train = pl.DataFrame({"label": ["low", "high", "low"]})
    assert make_label_map(train) == {"high": 0, "low": 1}
    with pytest.raises(ValueError, match="medium"):
        make_label_map(train, pl.DataFrame({"label": ["medium"]}), None)
