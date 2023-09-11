import json
from argparse import Namespace
from typing import Dict, Generator, List, Union

import torch
from infoquality.hyperparameters import HyperParameters


def batch_messages(
    messages: List[str], batch_size: int
) -> Generator[List[str], None, None]:
    for i in range(0, len(messages), batch_size):
        yield messages[i : i + batch_size]  # noqa


def save_hypers(
    params: Dict[str, Union[str, int, float]], output_dir: str, version: str
) -> str:
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


def model_size(model: torch.nn.Module) -> Dict[str, str]:
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
