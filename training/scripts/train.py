import json
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from infoquality.artifacts import bert_embeddings
from infoquality.hyperparameters import HyperParameters
from infoquality.preprocessor import Preprocessor
from infoquality.save import ModelSaver
from infoquality.utils import get_logger
from pydantic import BaseModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from infoquality.data import MessagesDataset
from infoquality.model import Model


def batch_messages(messages: List[str], batch_size: int) -> List[List[str]]:
    batches = []
    for i in range(0, len(messages), batch_size):
        batches.append(messages[i : i + batch_size])  # noqa
    return batches


def f1_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    largest_column = outputs.argmax(dim=1)

    f1s = []
    for i in range(outputs.shape[1]):
        if not any(largest_column == i):
            continue
        tp = ((largest_column == i) & (targets.int() == i)).sum().item()
        fp = ((largest_column == i) & (targets.int() != i)).sum().item()
        fn = ((largest_column != i) & (targets.int() == i)).sum().item()
        try:
            f1 = 2 * tp / (2 * tp + fp + fn)
            f1s.append(f1)
        except ZeroDivisionError:
            continue
    return sum(f1s) / len(f1s)


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    outputs_class = outputs.argmax(dim=1)
    acc = (outputs_class.int() == targets.int()).sum().item() / outputs.shape[0]
    return acc


class EpochMetrics(BaseModel):
    epoch: int
    loss: float
    val_loss: float
    val_acc: float
    val_f1: float


class Metrics:
    metrics: List[EpochMetrics] = []

    def collect(self) -> Dict[str, List[float]]:
        return {
            "epoch": self.epoch(),
            "loss": self.loss(),
            "val_loss": self.val_loss(),
            "val_acc": self.val_acc(),
            "val_f1": self.val_f1(),
        }

    def epoch(self) -> List[float]:
        return [float(m.epoch) for m in self.metrics]

    def loss(self) -> List[float]:
        return [m.loss for m in self.metrics]

    def val_loss(self) -> List[float]:
        return [m.val_loss for m in self.metrics]

    def val_acc(self) -> List[float]:
        return [m.val_acc for m in self.metrics]

    def val_f1(self) -> List[float]:
        return [m.val_f1 for m in self.metrics]

    def append(self, epoch: int, metrics: Dict[str, float]) -> None:
        epoch_metrics = EpochMetrics(epoch=epoch, **metrics)
        self.metrics.append(epoch_metrics)

    def save(
        self,
        output_dir: str,
        version: str = "",
    ) -> str:
        version = "" if version == "" else f"-{version}"
        save_as = f"{output_dir}/metrics{version}.json"
        data = self.collect()
        with open(save_as, "w") as f:
            json.dump(data, f, indent=2)
        return save_as


def save_hypers(params: dict, output_dir: str, version: str) -> str:
    version = "" if version == "" else f"-{version}"
    save_as = f"{output_dir}/hyperparameters{version}.json"
    with open(save_as, "w") as f:
        json.dump(params, f, indent=2)
    return save_as


def get_hyperparameters_from_args(args: Namespace) -> HyperParameters:
    d = args.__dict__
    kwargs = {
        k: v
        for k, v in d.items()
        if k in HyperParameters.model_fields and v is not None
    }
    return HyperParameters(**kwargs)


def get_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--num-epochs", type=int)
    parser.add_argument("--num-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--max-len", type=int)
    parser.add_argument("--embedding-dimensions", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--clip-value", type=float)
    parser.add_argument("--version", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--activation", type=str)
    parser.add_argument("--patience", type=int)
    return parser


def parse_args() -> Namespace:
    parser = get_parser()
    return parser.parse_args()


def main(args: Namespace):
    logger = get_logger(args.version, args.name)
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
        test_df = pl.read_parquet("/Users/mwk/data/movie-genre-prediction/test.parquet")

        label_map = {
            label: idx for idx, label in enumerate(train_df["label"].unique().sort())
        }

    logger.info("__nobs__", train=train_df.shape[0])
    logger.info("__nobs__", valid=valid_df.shape[0])
    logger.info("__nobs__", test=test_df.shape[0])
    train_data = MessagesDataset(
        messages=train_df["text"].to_list(),
        labels=train_df["label"].to_list(),
        label_map=label_map,
    )
    valid_data = MessagesDataset(
        messages=valid_df["text"].to_list(),
        labels=valid_df["label"].to_list(),
        label_map=label_map,
    )
    test_data = MessagesDataset(
        messages=test_df["text"].to_list(),
        labels=test_df["label"].to_list(),
        label_map=label_map,
    )

    # -------------------------------------------------------------------
    # HYPERPARAMETERS
    # -------------------------------------------------------------------
    hp = get_hyperparameters_from_args(args)
    for k, v in hp.__dict__.items():
        logger.info("__hp__", **{k: v})

    # -------------------------------------------------------------------
    # HYPERPARAMETERS & MODEL COMPONENTS
    # -------------------------------------------------------------------
    preprocessor = Preprocessor(max_len=hp.max_len)
    embeddings = (
        bert_embeddings[:, : hp.embedding_dimensions]  # noqa # type: ignore
        .detach()
        .clone()
    )
    model = Model(
        preprocessor=preprocessor,
        embeddings=embeddings,
        hyperparameters=hp,
        label_map=label_map,
    )
    optimizer = optim.AdamW(model.parameters(), lr=hp.lr)
    lr_scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=hp.gamma, patience=0, verbose=False
    )
    criterion = nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_data, batch_size=hp.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=hp.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=hp.batch_size, shuffle=True)

    # -------------------------------------------------------------------
    # SAVER
    # -------------------------------------------------------------------
    output_dir = Path("/Users/mwk/models/").joinpath(hp.name)
    # create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    saver = ModelSaver(str(output_dir), version=model.version)

    # -------------------------------------------------------------------
    # TRAINING LOOOP
    # -------------------------------------------------------------------
    metrics, best_metric = Metrics(), float("inf")
    best_epoch, best_state_dict = 0, model.state_dict()
    patience, early_stopping_counter = args.patience, 0
    start_time = datetime.now()

    for epoch in range(hp.num_epochs):
        trn_epoch_loss, val_epoch_loss = [], []
        val_epoch_acc, val_epoch_f1 = [], []
        epoch_lr = optimizer.param_groups[0]["lr"]

        # --------------------------------------------------------------
        # training steps
        # --------------------------------------------------------------
        model.train()

        for i, (messages, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(messages)
            loss = criterion(outputs, targets.long())
            loss.backward()
            nn.utils.clip_grad.clip_grad_value_(model.parameters(), hp.clip_value)
            optimizer.step()
            trn_epoch_loss.append(loss.item())
            if i == hp.num_steps:
                break

        # --------------------------------------------------------------
        # validation steps
        # --------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            for i, (vmessages, vtargets) in enumerate(valid_dataloader):
                voutputs = model(vmessages)
                vloss = criterion(voutputs, vtargets.long())
                val_epoch_loss.append(vloss.item())
                val_epoch_acc.append(accuracy(voutputs, vtargets))
                val_epoch_f1.append(f1_score(voutputs, vtargets))
                # if i == hp.num_steps:
                #     break

        # --------------------------------------------------------------
        # aggregate metrics
        # --------------------------------------------------------------
        trn_epoch_loss_stat = sum(trn_epoch_loss) / len(trn_epoch_loss)
        val_epoch_loss_stat = sum(val_epoch_loss) / len(val_epoch_loss)
        val_epoch_acc_stat = sum(val_epoch_acc) / len(val_epoch_acc)
        val_epoch_f1_stat = sum(val_epoch_f1) / len(val_epoch_f1)
        epoch_metrics = {
            "loss": trn_epoch_loss_stat,
            "val_loss": val_epoch_loss_stat,
            "val_acc": val_epoch_acc_stat,
            "val_f1": val_epoch_f1_stat,
        }
        metrics.append(epoch=epoch, metrics=epoch_metrics)
        # --------------------------------------------------------------
        # metrics logging
        # --------------------------------------------------------------
        epoch_metrics_logging = {
            "_loss": trn_epoch_loss_stat,
            "_val_loss": val_epoch_loss_stat,
            "val_acc": val_epoch_acc_stat,
            "val_f1": val_epoch_f1_stat,
        }
        epoch_metrics_logging = {
            k: f"{v:.4f}" for k, v in epoch_metrics_logging.items()
        }
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info(
            now, __ep=f"{epoch:3d}", __lr=f"{epoch_lr:.5f}", **epoch_metrics_logging
        )

        # --------------------------------------------------------------
        # track best epoch
        # --------------------------------------------------------------
        if val_epoch_loss_stat < best_metric:
            best_metric = val_epoch_loss_stat
            best_epoch = epoch
            best_state_dict = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                logger.info(
                    "__early_stopping__",
                    early_stopping_counter=early_stopping_counter,
                    patience=patience,
                )
                break

        lr_scheduler.step(val_epoch_loss_stat)

    # ------------------------------------------------------------------
    # training duration
    # ------------------------------------------------------------------
    end_time = datetime.now()
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
        epoch=best_epoch,
        metric="loss",
        value=f"{best_metric:.4f}",
    )

    # -------------------------------------------------------------------
    # TEST SET
    # -------------------------------------------------------------------
    with torch.no_grad():
        test_loss, acc, f1s = [], [], []
        for i, (tmessages, ttargets) in enumerate(test_dataloader):
            toutputs = model(tmessages)
            tloss = criterion(toutputs, ttargets.long())
            acc.append(accuracy(toutputs, ttargets))
            epoch_f1 = f1_score(toutputs, ttargets)
            f1s.append(epoch_f1)
            test_loss.append(tloss.item())
        tacc = sum(acc) / len(acc)
        tf1 = sum(f1s) / len(f1s)
        tlss = sum(test_loss) / len(test_loss)
        logger.info("test", loss=f"{tlss:.4f}", acc=f"{tacc:.4f}", f1=f"{tf1:.4f}")

    # ------------------------------------------------------------------
    # saving metadata
    # ------------------------------------------------------------------
    saved_as = metrics.save("/Users/mwk/models/meta", model.version)
    hyperparameters_path = save_hypers(
        params=model.hyperparameters.__dict__,
        output_dir="/Users/mwk/models/meta",
        version=model.version,
    )
    logger.info("__metadata__", hyperparameters=hyperparameters_path)
    logger.info("__metadata__", metrics=saved_as)

    # ------------------------------------------------------------------
    # HF DATASET SUBMISSION
    # ------------------------------------------------------------------
    if tacc > 0.33:
        logger.info("test_acc > 0.33 generating submission & saving model...")
        submission = pl.read_parquet(
            "/Users/mwk/data/movie-genre-prediction/submission-unlabeled.parquet"
        )
        msgs = batch_messages(submission["text"].to_list(), hp.batch_size)
        preds = []
        for batch in msgs:
            preds.extend(model(batch).argmax(1).tolist())
        revlabelmap = {v: k for k, v in label_map.items()}
        labels = pl.Series([revlabelmap[i] for i in preds])
        ids = pl.read_parquet(
            "/Users/mwk/data/movie-genre-prediction/submission-ids.parquet"
        )
        filename = datetime.now().strftime("submission-labeled-%Y%m%d%H%M%S.csv")
        ids.with_columns(genre=labels).write_csv(
            f"/Users/mwk/data/movie-genre-prediction/{filename}"
        )
        # -------------------------------------------------------------------
        # SAVE MODEL
        # -------------------------------------------------------------------
        model.load_state_dict(best_state_dict)
        model.eval()
        metrics_path = saver.save_metrics(metrics.__dict__)
        embeddings_path = saver.save_embeddings(model)
        state_dict_path = saver.save_state_dict(model)
        hyperparameters_path = saver.save_hyperparameters(model)
        logger.info("__save__", metrics=metrics_path)
        logger.info("__save__", embeddings=embeddings_path)
        logger.info("__save__", state_dict=state_dict_path)
        logger.info("__save__", hyperparameters=hyperparameters_path)

    else:
        logger.info("test_acc < 0.33 skipping submission & not saving model...")


if __name__ == "__main__":
    args = parse_args()
    main(args)
