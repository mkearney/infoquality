from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from common.evaluate import evaluate
from common.metrics import Fit, Metrics
from common.utils import (
    batch_messages,
    get_hyperparameters_from_args,
    log_metrics,
    model_size,
    save_hypers,
)
from datasets import load_dataset
from infoquality.save import ModelSaver
from infoquality.utils import get_logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from infoquality.data import InputsDataset
from infoquality.model import Model


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--best-metric", type=str)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--early-stopping-patience", type=int)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--lr-patience", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--model", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--num-dims", type=int)
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--version", type=str)
    return parser


def parse_args() -> Namespace:
    parser = get_parser()
    return parser.parse_args()


def main(args: Namespace):
    logger = get_logger(args.version, args.name)
    start_time = datetime.now()
    logger.info("_init_", time=start_time.strftime("%Y-%m-%d %H:%M:%S"))
    # -------------------------------------------------------------------
    # DATASETS: INFO QUALLITY
    # -------------------------------------------------------------------
    if False:
        train_df = pl.read_parquet("/Users/mwk/data/info-quality-train.parquet")
        valid_df = pl.read_parquet("/Users/mwk/data/info-quality-valid.parquet")
        test_df = pl.read_parquet("/Users/mwk/data/info-quality-test.parquet")
        label_map = {"low": 0, "high": 1}
    # -------------------------------------------------------------------
    # DATASETS: IMDB SENTIMENT
    # -------------------------------------------------------------------
    elif False:
        imdb = load_dataset("imdb")
        train_df = pl.DataFrame(
            {
                "text": imdb["train"]["text"],  # type: ignore
                "label": [
                    {0: "neg", 1: "pos"}[i]
                    for i in imdb["train"]["label"]  # type: ignore
                ],
            }
        )
        valid_df = pl.DataFrame(
            {
                "text": imdb["test"]["text"],  # type: ignore
                "label": [
                    {0: "neg", 1: "pos"}[i]
                    for i in imdb["test"]["label"]  # type: ignore
                ],
            }
        )
        valid_df = valid_df.sample(fraction=1, shuffle=True)
        valid_df = valid_df.filter(pl.col("text").str.strip().apply(len) > 0)
        splitrow = int(valid_df.shape[0] * 0.5)
        test_df = valid_df[splitrow:, :]  # noqa
        valid_df = valid_df[:splitrow, :]  # noqa
        label_map = {"neg": 0, "pos": 1}
    # -------------------------------------------------------------------
    # DATASETS: MOVIE GENRES (HUGGINGFACE)
    # -------------------------------------------------------------------
    else:
        train_df = pl.read_parquet(
            "/Users/mwk/data/movie-genre-prediction/train.parquet"
        )
        valid_df = pl.read_parquet(
            "/Users/mwk/data/movie-genre-prediction/valid.parquet"
        )
        # for train/valid - sample fraction (if < 1.0) subset
        if args.fraction < 1.0:
            train_df = train_df.sample(fraction=args.fraction, shuffle=True)
            valid_df = valid_df.sample(fraction=(args.fraction + 1) / 2, shuffle=True)

        # keep test set as is either way
        test_df = pl.read_parquet("/Users/mwk/data/movie-genre-prediction/test.parquet")

        # use sorted unique labels to create label map
        label_map = {
            label: idx for idx, label in enumerate(train_df["label"].unique().sort())
        }

    # -------------------------------------------------------------------
    # DATA SIZES, HYPERPARAMETERS, & TRAIN COMPONENTS
    # -------------------------------------------------------------------
    logger.info("_nobs_", train=train_df.shape[0])
    logger.info("_nobs_", valid=valid_df.shape[0])
    logger.info("_nobs_", test=test_df.shape[0])

    hp = get_hyperparameters_from_args(args)
    for k, v in hp.__dict__.items():
        logger.info("__hp__", **{k: v})

    model = Model(hyperparameters=hp)
    logger.info("_mdsz_", **model_size(model))
    optimizer = optim.AdamW(
        model.parameters(),  # type: ignore
        lr=hp.lr,
        eps=1e-8,
    )  # type: ignore
    lr_scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min" if hp.best_metric == "loss" else "max",
        factor=hp.gamma,
        patience=hp.lr_patience,
        verbose=False,
    )
    criterion = nn.CrossEntropyLoss()

    # preprocess data
    train_inputs = model.preprocess(train_df["text"].to_list())
    valid_inputs = model.preprocess(valid_df["text"].to_list())
    test_inputs = model.preprocess(test_df["text"].to_list())
    train_targets = torch.tensor(
        train_df["label"].map_dict(label_map).to_list(), dtype=torch.int64
    )
    valid_targets = torch.tensor(
        valid_df["label"].map_dict(label_map).to_list(), dtype=torch.int64
    )
    test_targets = torch.tensor(
        test_df["label"].map_dict(label_map).to_list(), dtype=torch.int64
    )
    # create datasets
    train_data = InputsDataset(
        input_ids=train_inputs["input_ids"],  # type: ignore
        attention_mask=train_inputs["attention_mask"],  # type: ignore
        targets=train_targets,
    )
    valid_data = InputsDataset(
        input_ids=valid_inputs["input_ids"],  # type: ignore
        attention_mask=valid_inputs["attention_mask"],  # type: ignore
        targets=valid_targets,
    )
    test_data = InputsDataset(
        input_ids=test_inputs["input_ids"],  # type: ignore
        attention_mask=test_inputs["attention_mask"],  # type: ignore
        targets=test_targets,
    )
    train_dataloader = DataLoader(train_data, batch_size=hp.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=hp.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=hp.batch_size, shuffle=True)
    fit = Fit(num_classes=hp.num_classes)

    # -------------------------------------------------------------------
    # SAVER
    # -------------------------------------------------------------------
    output_dir = Path("/Users/mwk/models/").joinpath(hp.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = ModelSaver(
        str(output_dir), version=model.version, logger=logger
    )  # type: ignore

    # -------------------------------------------------------------------
    # TRAINING LOOOP
    # -------------------------------------------------------------------
    metrics = Metrics()
    best_metric_value, best_alt_value = float("inf"), 0
    best_metric_value *= 1 if hp.best_metric == "loss" else -1
    best_epoch, best_state_dict = 0, model.state_dict()  # type: ignore
    es_counter = 0
    tacc, val_epoch_acc_stat = 0.0, 0.0

    for epoch in range(hp.num_epochs):
        epoch_lr = optimizer.param_groups[0]["lr"]

        # --------------------------------------------------------------
        # training steps
        # --------------------------------------------------------------
        trn_epoch_loss = []
        model.train()  # type: ignore
        for i, data in enumerate(train_dataloader):
            outputs = model(**data)
            loss = criterion(outputs, data["targets"].long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            trn_epoch_loss.append(loss.item())
            if i == hp.num_steps:
                break
        trn_epoch_loss_stat = sum(trn_epoch_loss) / len(trn_epoch_loss)
        # --------------------------------------------------------------
        # validation steps
        # --------------------------------------------------------------
        epoch_metrics = evaluate(
            model=model,
            dataloader=valid_dataloader,
            criterion=criterion,
            train_loss=trn_epoch_loss_stat,
        )
        metrics.append(epoch=epoch, metrics=epoch_metrics)
        log_metrics(epoch, epoch_lr, epoch_metrics, logger)
        # --------------------------------------------------------------
        # track best metric
        # --------------------------------------------------------------
        if hp.best_metric == "loss" and epoch_metrics["val_loss"] < best_metric_value:
            best_metric_value = epoch_metrics["val_loss"]
            best_alt_value = epoch_metrics["val_acc"]
            best_epoch = epoch
            best_state_dict = model.state_dict()  # type: ignore
            es_counter = 0
        elif epoch_metrics["val_acc"] > best_metric_value:
            best_metric_value = epoch_metrics["val_acc"]
            best_alt_value = epoch_metrics["val_loss"]
            best_epoch = epoch
            best_state_dict = model.state_dict()  # type: ignore
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= hp.early_stopping_patience:
                logger.info("__early_stopping__", es_counter=es_counter)
                break
        if hp.best_metric == "loss":
            lr_scheduler.step(epoch_metrics["val_loss"])
        else:
            lr_scheduler.step(epoch_metrics["val_acc"])

    # ------------------------------------------------------------------
    # training duration
    # ------------------------------------------------------------------
    end_time = datetime.now()
    logger.info("__end__", time=end_time.strftime("%Y-%m-%d %H:%M:%S"))
    duration = end_time - start_time
    if duration.total_seconds() <= 120:
        logger.info("duration", seconds=f"{duration.total_seconds():,.2f}")
    else:
        logger.info("duration", minutes=f"{duration.total_seconds()/60:,.2f}")
    # ------------------------------------------------------------------
    # best epoch logging
    # ------------------------------------------------------------------
    logger.info(
        "best_metric",
        _epoch=best_epoch,
        _metric=hp.best_metric,
        _value=f"{best_metric_value:.4f}",
        alt=f"{best_alt_value:.4f}",
    )

    # -------------------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------------------
    model.load_state_dict(best_state_dict)  # type: ignore
    model.eval()  # type: ignore
    saver.save(model, metrics.__dict__)

    # -------------------------------------------------------------------
    # TEST SET
    # -------------------------------------------------------------------
    if val_epoch_acc_stat > 0.2:
        with torch.no_grad():
            test_loss, acc, f1s, prs, rcs = [], [], [], [], []
            for data in test_dataloader:
                outputs = model(**data)  # type: ignore
                loss = criterion(outputs, data["targets"].long())
                fit_metrics = fit(outputs, data["targets"])
                acc.append(fit_metrics.acc)
                f1s.append(fit_metrics.f1)
                prs.append(fit_metrics.pr)
                rcs.append(fit_metrics.rc)
                test_loss.append(loss.item())
            denom = len(acc)
            tacc = sum(acc) / denom
            tf1 = sum(f1s) / denom
            tlss = sum(test_loss) / denom
            tpr = sum(prs) / denom
            trc = sum(rcs) / denom
            logger.info(
                "test",
                loss=f"{tlss:.4f}",
                acc=f"{tacc:.4f}",
                f1=f"{tf1:.4f}",
                pr=f"{tpr:.4f}",
                rc=f"{trc:.4f}",
            )

    # ------------------------------------------------------------------
    # saving metadata
    # ------------------------------------------------------------------
    saved_as = metrics.save(
        "/Users/mwk/models/meta",
        model.version,  # type: ignore
    )
    hyperparameters_path = save_hypers(
        params=model.hyperparameters.__dict__,
        output_dir="/Users/mwk/models/meta",
        version=model.version,  # type: ignore
    )
    logger.info("__metadata__", hyperparameters=hyperparameters_path)
    logger.info("__metadata__", metrics=saved_as)

    # ------------------------------------------------------------------
    # HF DATASET SUBMISSION
    # ------------------------------------------------------------------
    if tacc > 0.39:
        logger.info("test_acc > 0.39 generating submission & saving model...")
        submission = pl.read_parquet(
            "/Users/mwk/data/movie-genre-prediction/submission-unlabeled.parquet"
        )
        msgs = batch_messages(submission["text"].to_list(), hp.batch_size)
        revlabelmap = {v: k for k, v in label_map.items()}
        preds = []
        with torch.no_grad():
            for batch in msgs:
                preds.extend([revlabelmap[i.item()] for i in model(batch).argmax(1)])
            labels = pl.Series(preds)
            ids = pl.read_parquet(
                "/Users/mwk/data/movie-genre-prediction/submission-ids.parquet"
            )
            filename = datetime.now().strftime("submission-labeled-%Y%m%d%H%M%S.csv")
            ids.with_columns(genre=labels).write_csv(
                f"/Users/mwk/data/movie-genre-prediction/{filename}"
            )

    else:
        logger.info("test_acc < 0.39 skipping submission & not saving model...")


if __name__ == "__main__":
    args = parse_args()
    main(args)
