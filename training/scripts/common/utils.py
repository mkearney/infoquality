from argparse import Namespace

import torch

from infoquality.hyperparameters import HyperParameters


def get_hyperparameters_from_args(args: Namespace) -> HyperParameters:
    """
    Get hyperparameters from command line arguments

    ### Args:
        - `args` Command line arguments as returned by argparse

    ### Returns:
        - An instance of HyperParameters (defaults for unset arguments)
    """
    kwargs = {
        k: v
        for k, v in vars(args).items()
        if k in HyperParameters.model_fields and v is not None
    }
    return HyperParameters(**kwargs)


def model_size(model: torch.nn.Module) -> dict[str, str]:
    """
    Returns information about the size of a model

    ### Args:
        - `model` nn.Module of interest

    ### Returns:
        - dictionary with "parameters" (the number of parameters) and
            "memory" (the amount of memory used in MB)
    """
    tensors = [*model.parameters(), *model.buffers()]
    count = sum(t.numel() for t in tensors)
    memory = sum(t.numel() * t.element_size() for t in tensors) / 1024**2
    return {
        "parameters": f"{count:,}",
        "memory": f"{memory:,.1f} MB",
    }


def log_metrics(epoch: int, epoch_lr: float, metrics: dict[str, float], logger) -> None:
    epoch_metrics_logging = {
        "_trn": metrics["loss"],
        "_val": metrics["val_loss"],
        "acc": metrics["val_acc"],
        "f1": metrics["val_f1"],
        "pr": metrics["val_pr"],
        "rc": metrics["val_rc"],
    }
    epoch_metrics_logging = {k: f"{v:.4f}" for k, v in epoch_metrics_logging.items()}
    epp = f"{epoch:2d}"
    epp = f"{epp:^6}".replace(" ", "_")
    logger.info(f"{epp}", __lr=f"{epoch_lr:.5f}", **epoch_metrics_logging)


def format_duration(seconds: float) -> dict[str, str]:
    if seconds <= 120:
        return {"seconds": f"{seconds:,.2f}"}
    return {"minutes": f"{seconds / 60:,.2f}"}
