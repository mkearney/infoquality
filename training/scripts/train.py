"""
Fine-tune a pretrained transformer for text classification.

Examples:

    # your own data (.parquet/.csv/.tsv/.json/.jsonl with text + label columns)
    python training/scripts/train.py \
        --train-path data/train.parquet \
        --valid-path data/valid.parquet \
        --test-path data/test.parquet \
        --name infoquality

    # IMDB sentiment from the Hugging Face hub
    python training/scripts/train.py --dataset imdb --name imdbsent
"""

import os
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from common.data import DATASETS, encode, load_splits, make_label_map, subsample
from common.evaluate import evaluate
from common.metrics import Metrics
from common.training import EarlyStopping, select_device, set_seed, train_one_epoch
from common.utils import (
    format_duration,
    get_hyperparameters_from_args,
    log_metrics,
    model_size,
)
from infoquality.hyperparameters import HyperParameters
from infoquality.model import Model
from infoquality.save import ModelSaver
from infoquality.utils import get_logger

DEFAULT_OUTPUT_DIR = os.environ.get("INFOQUALITY_OUTPUT_DIR", "models")


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Fine-tune a text classification model.")
    data = parser.add_argument_group("data")
    data.add_argument("--dataset", choices=DATASETS, default="files")
    data.add_argument("--train-path", type=str, help="training data file")
    data.add_argument(
        "--valid-path",
        type=str,
        help="validation data file (default: hold out --valid-size of train)",
    )
    data.add_argument(
        "--test-path", type=str, help="test data file (default: no test set)"
    )
    data.add_argument("--text-col", type=str, default="text")
    data.add_argument("--label-col", type=str, default="label")
    data.add_argument("--valid-size", type=float, default=0.1)
    data.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="subsample this fraction of train and valid (test is kept as is)",
    )

    run = parser.add_argument_group("run")
    run.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="models are saved to {output-dir}/{name}/{version} "
        "(default: $INFOQUALITY_OUTPUT_DIR or ./models)",
    )
    run.add_argument("--device", type=str, default="auto")

    hp = parser.add_argument_group("hyperparameters (defaults: HyperParameters)")
    for name, field in HyperParameters.model_fields.items():
        arg_type = field.annotation if field.annotation in (int, float) else str
        hp.add_argument(
            f"--{name.replace('_', '-')}",
            type=arg_type,
            help=f"default: {field.default}",
        )
    return parser


def parse_args(argv: list[str] | None = None) -> Namespace:
    return get_parser().parse_args(argv)


def main(args: Namespace) -> dict[str, Any]:
    logger = get_logger()
    start_time = datetime.now()
    logger.info("_init_", time=start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # -------------------------------------------------------------------
    # HYPERPARAMETERS & DATA
    # -------------------------------------------------------------------
    hp = get_hyperparameters_from_args(args)
    set_seed(hp.seed)
    device = select_device(args.device)

    train_df, valid_df, test_df = load_splits(
        dataset=args.dataset,
        train_path=args.train_path,
        valid_path=args.valid_path,
        test_path=args.test_path,
        text_col=args.text_col,
        label_col=args.label_col,
        valid_size=args.valid_size,
        seed=hp.seed,
    )
    train_df = subsample(train_df, args.fraction, hp.seed)
    valid_df = subsample(valid_df, args.fraction, hp.seed)
    label_map = make_label_map(train_df, valid_df, test_df)
    if args.num_classes is not None and args.num_classes != len(label_map):
        raise ValueError(
            f"--num-classes={args.num_classes} but the training data has "
            f"{len(label_map)} labels: {list(label_map)}"
        )
    hp = hp.model_copy(update={"num_classes": len(label_map)})

    logger.info("_nobs_", train=train_df.height)
    logger.info("_nobs_", valid=valid_df.height)
    logger.info("_nobs_", test=0 if test_df is None else test_df.height)
    logger.info("_labs_", labels=list(label_map))
    for k, v in hp.model_dump().items():
        logger.info("__hp__", **{k: v})

    # -------------------------------------------------------------------
    # MODEL & TRAIN COMPONENTS
    # -------------------------------------------------------------------
    model = Model(hyperparameters=hp, label_map=label_map)
    logger.info("_mdsz_", **model_size(model))
    model.to(device)
    logger.info("_devc_", device=str(device))

    # create the output directory before training so a bad path fails fast
    saver = ModelSaver(
        Path(args.output_dir) / hp.name, version=model.version, logger=logger
    )

    optimizer = optim.AdamW(model.parameters(), lr=hp.lr)
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min" if hp.best_metric == "loss" else "max",
        factor=hp.gamma,
        patience=hp.lr_patience,
    )
    criterion = nn.CrossEntropyLoss()

    generator = torch.Generator().manual_seed(hp.seed)
    train_dataloader = DataLoader(
        encode(model, train_df, label_map),
        batch_size=hp.batch_size,
        shuffle=True,
        generator=generator,
    )
    valid_dataloader = DataLoader(
        encode(model, valid_df, label_map), batch_size=hp.batch_size
    )

    # -------------------------------------------------------------------
    # TRAINING LOOP
    # -------------------------------------------------------------------
    metrics = Metrics()
    early_stopping = EarlyStopping(hp.best_metric, hp.early_stopping_patience)

    for epoch in range(hp.num_epochs):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(
            model,
            train_dataloader,
            optimizer,
            criterion,
            device=device,
            max_steps=hp.num_steps,
        )
        valid = evaluate(
            model, valid_dataloader, criterion, hp.num_classes, device=device
        )
        epoch_metrics = {
            "loss": train_loss,
            **{f"val_{k}": v for k, v in valid.items()},
        }
        metrics.append(epoch=epoch, lr=epoch_lr, metrics=epoch_metrics)
        log_metrics(epoch, epoch_lr, epoch_metrics, logger)

        early_stopping.step(epoch, epoch_metrics, model)
        if early_stopping.should_stop:
            logger.info("__early_stopping__", es_counter=early_stopping.counter)
            break
        lr_scheduler.step(epoch_metrics[early_stopping.key])

    end_time = datetime.now()
    logger.info("__end__", time=end_time.strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("duration", **format_duration((end_time - start_time).total_seconds()))
    logger.info(
        "best_metric",
        _epoch=early_stopping.best_epoch,
        _metric=hp.best_metric,
        _value=f"{early_stopping.best_value:.4f}",
        alt=f"{early_stopping.best_alt_value:.4f}",
    )

    # -------------------------------------------------------------------
    # RESTORE BEST WEIGHTS, EVALUATE ON TEST SET, & SAVE
    # -------------------------------------------------------------------
    if early_stopping.best_state_dict is not None:
        model.load_state_dict(early_stopping.best_state_dict)
    model.eval()

    results: dict[str, Any] = {
        "epochs": metrics.collect(),
        "best_epoch": early_stopping.best_epoch,
    }
    if test_df is not None:
        test_dataloader = DataLoader(
            encode(model, test_df, label_map), batch_size=hp.batch_size
        )
        test = evaluate(
            model, test_dataloader, criterion, hp.num_classes, device=device
        )
        logger.info("test", **{k: f"{v:.4f}" for k, v in test.items()})
        results["test"] = test

    saver.save(model, results)
    results["path"] = str(saver.path)
    return results


if __name__ == "__main__":
    main(parse_args())
