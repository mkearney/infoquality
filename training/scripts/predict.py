"""
Predict labels for a file of texts with a model saved by train.py.

Example:

    python training/scripts/predict.py \
        --model-dir models/infoquality/0.1.0.20260101120000 \
        --input data/unlabeled.parquet \
        --output predictions.csv \
        --keep-cols id
"""

from argparse import ArgumentParser, Namespace

import polars as pl

from common.data import read_table
from common.training import select_device
from infoquality.predictor import Predictor


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(description="Predict labels for a file of texts.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--input", required=True, help=".parquet/.csv/.jsonl file")
    parser.add_argument("--output", required=True, help=".csv or .parquet file")
    parser.add_argument("--text-col", default="text")
    parser.add_argument(
        "--keep-cols",
        nargs="*",
        default=[],
        help="input columns to copy to the output (e.g. ids)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--proba", action="store_true", help="also write class probabilities"
    )
    return parser.parse_args(argv)


def main(args: Namespace) -> pl.DataFrame:
    df = read_table(args.input)
    predictor = Predictor(args.model_dir, device=str(select_device(args.device)))
    texts = df[args.text_col].cast(pl.String).fill_null("").to_list()
    probas = predictor.predict_proba(texts, batch_size=args.batch_size)
    out = df.select(args.keep_cols).with_columns(
        label=pl.Series([predictor.labels[i] for i in probas.argmax(1).tolist()])
    )
    if args.proba:
        out = out.with_columns(
            pl.Series(f"proba_{label}", probas[:, i].tolist())
            for i, label in enumerate(predictor.labels)
        )
    if args.output.endswith(".parquet"):
        out.write_parquet(args.output)
    else:
        out.write_csv(args.output)
    return out


if __name__ == "__main__":
    main(parse_args())
