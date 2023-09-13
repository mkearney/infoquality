import json
from argparse import Namespace
from typing import Dict, Generator, List, Union

import torch
from infoquality.hyperparameters import HyperParameters


def batch_messages(
    messages: List[str], batch_size: int
) -> Generator[List[str], None, None]:
    """
    Splits list of messages into batches of given size

    ### Args:
        - `messages`: A list of messages to split
        - `batch_size`: Size of each batch

    ### Returns:
        - Generator of lists of messages each is the size of batch_size
    """
    for i in range(0, len(messages), batch_size):
        yield messages[i : i + batch_size]  # noqa


def save_hypers(
    params: Dict[str, Union[str, int, float]],
    output_dir: str,
    version: str,
) -> str:
    """
    Save hyperparameters to json file

    ### Args:
        - `params` A dictionary of parameters that will be saved
        - `output_dir` The directory to save the model to
        - `version` The version of the model to save

    ### Returns:
        - The name of the file
    """
    version = "" if version == "" else f"-{version}"
    save_as = f"{output_dir}/hyperparameters{version}.json"
    with open(save_as, "w") as f:
        json.dump(params, f, indent=2)
    return save_as


def get_hyperparameters_from_args(args: Namespace) -> HyperParameters:
    """
    Get hyperparameters from command line arguments

    ### Args:
        - `args` Command line arguments as returned by argparse

    ### Returns:
        - An instance of HyperParameters
    """
    d = args.__dict__
    kwargs = {
        k: v
        for k, v in d.items()
        if k in HyperParameters.model_fields and v is not None
    }
    return HyperParameters(**kwargs)


def model_size(model: torch.nn.Module) -> Dict[str, str]:
    """
    Returns information about the size of a model

    ### Args:
        - `model` nn.Module of interest

    ### Returns:
        - dictionary with "parameters" (the number of parameters) and
            "memory" (the amount of memory used in MB)
    """
    param_count = sum(param.numel() for param in model.parameters())
    param_size = sum(
        param.nelement() * param.element_size() for param in model.parameters()
    )
    buffer_count = sum(buffer.numel() for buffer in model.buffers())
    buffer_size = sum(
        buffer.nelement() * buffer.element_size() for buffer in model.buffers()
    )
    memory = (param_size + buffer_size) / 1024**2
    parameters = param_count + buffer_count
    return {
        "parameters": f"{parameters:,}",
        "memory": f"{memory:,.1f} MB",
    }


def log_metrics(epoch: int, epoch_lr: float, metrics: Dict[str, float], logger) -> None:
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
