from argparse import ArgumentParser, Namespace
from datetime import datetime

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from infoquality.artifacts import bert_embeddings
from infoquality.data import MessagesDataset
from infoquality.hyperparameters import HyperParameters
from infoquality.model import Model
from infoquality.preprocessor import Preprocessor
from infoquality.save import ModelSaver


def f1(outputs, targets) -> float:
    tp = (outputs.argmax(1) == targets.int()).sum().item()
    fp = (outputs.argmax(1) != targets.int()).sum().item()
    fn = (outputs.argmax(1) == targets.int()).sum().item()
    f1 = 2 * tp / (2 * tp + fp + fn)
    return f1


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
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--version", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--output-dir", type=str, default="/Users/mwk/models/iqmodel")
    return parser


def parse_args() -> Namespace:
    parser = get_parser()
    return parser.parse_args()


def main(args: Namespace):
    print("__start__")

    # -------------------------------------------------------------------
    # GET DATASETS
    # -------------------------------------------------------------------
    train_df = pl.read_parquet("/Users/mwk/data/info-quality-train.parquet")
    valid_df = pl.read_parquet("/Users/mwk/data/info-quality-valid.parquet")
    test_df = pl.read_parquet("/Users/mwk/data/info-quality-test.parquet")
    label_map = {"low": 0, "high": 1}
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
    # test_data = MessagesDataset(
    #     messages=test_df["text"].to_list(),
    #     labels=test_df["label"].to_list(),
    #     label_map=label_map,
    # )

    # -------------------------------------------------------------------
    # HYPERPARAMETERS & MODEL COMPONENTS
    # -------------------------------------------------------------------
    hp = get_hyperparameters_from_args(args)
    preprocessor = Preprocessor(max_len=hp.max_len)
    embeddings = bert_embeddings[:, : hp.embedding_dim].detach().clone()  # type: ignore
    model = Model(
        preprocessor=preprocessor,
        embeddings=embeddings,
        hyperparameters=hp,
    )
    optimizer = optim.AdamW(model.parameters(), lr=hp.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=hp.num_steps,
        gamma=hp.gamma,
    )
    criterion = nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_data, batch_size=hp.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=hp.batch_size, shuffle=True)

    # -------------------------------------------------------------------
    # SAVER
    # -------------------------------------------------------------------
    saver = ModelSaver(args.output_dir, version=model.version)

    # -------------------------------------------------------------------
    # TRAINING LOOOP
    # -------------------------------------------------------------------
    best_metric, best_state_dict = 0, model.state_dict()
    valid_loss, acc, f1s = 0, [], []
    total_loss = 0

    for epoch in range(hp.num_epochs):
        if epoch == 0:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now}")
            print("-" * 73)
            print(
                f"{'time':>19} {'ep':>2} {'lr0e1':>7} {'lr1e1':>7}  {'loss':>6}  "
                f" {'val':>6}   {'acc':>6}   {'f1':>6}"
            )
            print("-" * 73)
        total_loss = 0
        model.train()
        lr0 = optimizer.param_groups[0]["lr"]
        lr1 = lr0
        for i, (messages, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(messages)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            lr1 = optimizer.param_groups[0]["lr"]
            total_loss += loss.item() / len(targets)
            if i == hp.num_steps:
                break

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss, acc, f1s = 0, [], []
            for i, (vmessages, vtargets) in enumerate(valid_dataloader):
                voutputs = model(vmessages)
                vloss = criterion(voutputs, vtargets)
                acc.append(
                    ((voutputs.argmax(1)).int() == vtargets.int()).sum().item()
                    / voutputs.shape[0]
                )
                epoch_f1 = f1(voutputs, vtargets)
                f1s.append(epoch_f1)
                if epoch_f1 > best_metric:
                    best_metric = epoch_f1
                    best_state_dict = model.state_dict()

                valid_loss += vloss.item() / len(vtargets)
                if i == hp.num_steps:
                    break

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        vacc = sum(acc) / len(acc)
        vf1 = sum(f1s) / len(f1s)
        print(
            f"{now} {epoch:2d} {lr0:6.5f} {lr1:6.5f}  {total_loss:6.4f}  "
            f" {valid_loss:6.4f}   {vacc:4.4f}   {vf1:4.4f}"
        )
        if epoch == hp.num_epochs - 1:
            print("-" * 73)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{now}")

    model.load_state_dict(best_state_dict)
    model.eval()
    metrics = {
        "loss": total_loss,
        "f1": f1s,
        "acc": acc,
        "val": valid_loss,
    }
    metrics_path = saver.save_metrics(metrics)
    print(f"Model metrics saved as {metrics_path}")
    embeddings_path = saver.save_embeddings(model)
    print(f"Model saved as {embeddings_path}")
    state_dict_path = saver.save_state_dict(model)
    print(f"Model state dict saved as {state_dict_path}")
    hyperparameters_path = saver.save_hyperparameters(model)
    print(f"Model hyperparameters saved as {hyperparameters_path}")

    test_high = test_df.filter(pl.col("label") == "high")["text"].head(200).to_list()
    test_low = test_df.filter(pl.col("label") == "low")["text"].head(200).to_list()

    # revlabelmap = {v: k for k, v in label_map.items()}
    pl.Config.set_fmt_str_lengths(140)
    pl.Config.set_tbl_rows(40)

    pred_est = torch.sigmoid(model(test_high + test_low))[:, 1].tolist()
    pred_class = ["high" if i > 0.5 else "low" for i in pred_est]

    df = pl.DataFrame(
        {
            "text": test_high + test_low,
            "pred": pred_est,
            "class": pred_class,
            "label": ["high"] * len(test_high) + ["low"] * len(test_low),
        }
    ).sort("pred")

    test_acc = (
        df.with_columns(y=pl.col("label") == pl.col("class"))["y"].sum() / df.shape[0]
    )
    print(f"Test accuracy: {test_acc*100:.2f}%")


if __name__ == "__main__":
    args = parse_args()
    main(args)
