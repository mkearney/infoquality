import random

import torch
from torch import nn
from torch.utils.data import DataLoader


def select_device(name: str = "auto") -> torch.device:
    """Return `name` as a device, or for "auto", the best available device."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device | str = "cpu",
    max_steps: int = 0,
) -> float:
    """
    Train for (at most `max_steps` batches of) one epoch.

    ### Returns
        - mean training loss over the batches seen
    """
    model.train()
    losses = []
    for i, data in enumerate(dataloader):
        if max_steps > 0 and i >= max_steps:
            break
        data = {k: v.to(device) for k, v in data.items()}
        outputs = model(**data)
        loss = criterion(outputs, data["targets"])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    if not losses:
        raise ValueError("no training batches")
    return sum(losses) / len(losses)


class EarlyStopping:
    """
    Tracks the best epoch (keeping a copy of its weights) and signals when
    training should stop.

    ### Args
        - `best_metric`: "loss" (validation loss, lower is better) or "acc"
            (validation accuracy, higher is better)
        - `patience`: stop after this many consecutive epochs without
            improvement
    """

    def __init__(self, best_metric: str = "loss", patience: int = 4):
        if best_metric not in ("loss", "acc"):
            raise ValueError("best_metric must be 'loss' or 'acc'")
        self.key = f"val_{best_metric}"
        self.alt_key = "val_acc" if best_metric == "loss" else "val_loss"
        self.mode = "min" if best_metric == "loss" else "max"
        self.patience = patience
        self.best_value = float("inf") if self.mode == "min" else float("-inf")
        self.best_alt_value = float("nan")
        self.best_epoch = -1
        self.best_state_dict: dict[str, torch.Tensor] | None = None
        self.counter = 0

    def is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def step(self, epoch: int, metrics: dict[str, float], model: nn.Module) -> bool:
        """Record an epoch's metrics; returns True if it is the new best."""
        value = metrics[self.key]
        if self.is_improvement(value):
            self.best_value = value
            self.best_alt_value = metrics[self.alt_key]
            self.best_epoch = epoch
            # copy: state_dict() returns references to the live parameters
            self.best_state_dict = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            self.counter = 0
            return True
        self.counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.counter >= self.patience
