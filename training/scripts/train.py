from argparse import ArgumentParser, Namespace
from datetime import datetime

import polars as pl
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from infoquality.artifacts import bert_embeddings
from infoquality.hyperparameters import HyperParameters
from infoquality.preprocessor import Preprocessor
from infoquality.save import ModelSaver
from torch.utils.data import DataLoader

from infoquality.data import MessagesDataset
from infoquality.model import Model

# def f1(outputs, targets) -> float:
#     tp = (outputs.argmax(1) == targets.int()).sum().item()
#     fp = (outputs.argmax(1) != targets.int()).sum().item()
#     fn = (outputs.argmax(1) == targets.int()).sum().item()
#     f1 = 2 * tp / (2 * tp + fp + fn)
#     return f1


def f1_score(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    if outputs.shape[1] <= 2:
        largest_column = torch.sigmoid(outputs)[:, 1] > 0.5
    else:
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
    if outputs.shape[1] <= 2:
        outputs_class = torch.sigmoid(outputs)[:, 1] > 0.5
    else:
        outputs_class = outputs.argmax(dim=1)
    acc = (outputs_class.int() == targets.int()).sum().item() / outputs.shape[0]
    return acc


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
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument(
        "--output-dir", type=str, default="/Users/mwk/models/moviegenre"
    )
    return parser


def parse_args() -> Namespace:
    parser = get_parser()
    return parser.parse_args()


def main(args: Namespace):
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
        test_df = valid_df[splitrow:, :]
        valid_df = valid_df[:splitrow, :]
        label_map = {"neg": 0, "pos": 1}
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
    print("-" * 76)
    print(f"         train nobs:  {train_df.shape[0]:,}")
    print(f"         valid nobs:  {valid_df.shape[0]:,}")
    print(f"          test nobs:  {test_df.shape[0]:,}")
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
    print("-" * 76)
    hp = get_hyperparameters_from_args(args)
    for k, v in hp.__dict__.items():
        print(f" {k:>18}:  {v}")

    # -------------------------------------------------------------------
    # HYPERPARAMETERS & MODEL COMPONENTS
    # -------------------------------------------------------------------
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
        step_size=int(hp.num_steps * args.theta),
        gamma=hp.gamma,
    )
    criterion = nn.CrossEntropyLoss()
    train_dataloader = DataLoader(train_data, batch_size=hp.batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_data, batch_size=hp.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=hp.batch_size, shuffle=True)

    # -------------------------------------------------------------------
    # SAVER
    # -------------------------------------------------------------------
    saver = ModelSaver(args.output_dir, version=model.version)

    # -------------------------------------------------------------------
    # TRAINING LOOOP
    # -------------------------------------------------------------------
    best_metric, best_state_dict = 99, model.state_dict()
    valid_loss, acc, f1s = 0, [], []
    total_loss = 0
    start_time = datetime.now()
    for epoch in range(hp.num_epochs):
        if epoch == 0:
            print("-" * 76)
            print(
                f" {'time':>19}  {'ep':>2} {'lr0e1':>7} {'lr1e1':>7}  {'loss':>6}  "
                f" {'val':>6}   {'acc':>6}   {'f1':>6}"
            )
            print("-" * 76)
        train_loss = []
        model.train()
        lr0 = optimizer.param_groups[0]["lr"]
        lr1 = lr0
        for i, (messages, targets) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(messages)
            loss = criterion(outputs, targets.long())
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            lr1 = optimizer.param_groups[0]["lr"]
            train_loss.append(loss.item() / len(targets))
            if i == hp.num_steps:
                break

        # Validation
        model.eval()
        with torch.no_grad():
            valid_loss, acc, f1s = [], [], []
            for i, (vmessages, vtargets) in enumerate(valid_dataloader):
                voutputs = model(vmessages)
                vloss = criterion(voutputs, vtargets.long())
                voutputs = torch.softmax(voutputs, dim=-1)
                acc.append(accuracy(voutputs, vtargets))
                epoch_f1 = f1_score(voutputs, vtargets)
                f1s.append(epoch_f1)
                if vloss < best_metric:
                    best_metric = vloss
                    best_state_dict = model.state_dict()

                valid_loss.append(vloss.item() / len(vtargets))
                if i == hp.num_steps:
                    break

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tlss = sum(train_loss) / (len(train_loss) - 1)
        vacc = sum(acc) / (len(acc) - 1)
        vf1 = sum(f1s) / (len(f1s) - 1)
        vlss = sum(valid_loss) / (len(valid_loss) - 1)
        print(
            f" {now}  {epoch:2d} {lr0:6.5f} {lr1:6.5f}  {tlss:6.4f}  "
            f" {vlss:6.4f}   {vacc:4.4f}   {vf1:4.4f}"
        )
        if epoch == hp.num_epochs - 1:
            print("-" * 76)
            end_time = datetime.now()
            duration = end_time - start_time
            if duration.total_seconds() <= 120:
                print(f"     train_duration:  {duration.total_seconds():,.2f} seconds")
            else:
                print(
                    f"     train_duration:  {duration.total_seconds()/60:,.2f} minutes"
                )
    # -------------------------------------------------------------------
    # TEST
    # -------------------------------------------------------------------
    with torch.no_grad():
        test_loss, acc, f1s = [], [], []
        for i, (tmessages, ttargets) in enumerate(test_dataloader):
            toutputs = model(tmessages)
            tloss = criterion(toutputs, ttargets.long())
            toutputs = torch.softmax(toutputs, dim=-1)
            acc.append(accuracy(toutputs, ttargets))
            epoch_f1 = f1_score(toutputs, ttargets)
            f1s.append(epoch_f1)
            test_loss.append(tloss.item() / len(ttargets))
            if i == hp.num_steps:
                break
        print("-" * 76)
        tacc = sum(acc) / len(acc)
        tf1 = sum(f1s) / len(f1s)
        tlss = sum(test_loss) / len(test_loss)
        print(f"          test_loss:  {tlss:.4f}")
        print(f"           test_acc:  {tacc:.4f}")
        print(f"            test_f1:  {tf1:.4f}")

    # -------------------------------------------------------------------
    # SAVE MODEL
    # -------------------------------------------------------------------
    print("-" * 76)
    model.load_state_dict(best_state_dict)
    model.eval()
    metrics = {
        "loss": total_loss,
        "f1": f1s,
        "acc": acc,
        "val": valid_loss,
    }
    metrics_path = saver.save_metrics(metrics)
    print(f"            metrics:  {metrics_path}")
    embeddings_path = saver.save_embeddings(model)
    print(f"         embeddings:  {embeddings_path}")
    state_dict_path = saver.save_state_dict(model)
    print(f"         state_dict:  {state_dict_path}")
    hyperparameters_path = saver.save_hyperparameters(model)
    print(f"    hyperparameters:  {hyperparameters_path}")
    print("-" * 76)


if __name__ == "__main__":
    args = parse_args()
    main(args)
