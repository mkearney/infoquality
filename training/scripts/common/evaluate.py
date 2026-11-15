import torch
from torch import nn
from torch.utils.data import DataLoader

from common.metrics import Fit


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """
    Evaluate a model on every batch of `dataloader`.

    ### Args
        - `model`: model
        - `dataloader`: dataloader
        - `criterion`: loss function
        - `num_classes`: number of classes
        - `device`: device to run on

    ### Returns
        - dict with `loss` (mean per example), `acc`, `f1`, `pr`, and `rc`
    """
    fit = Fit(num_classes=num_classes).to(device)
    total_loss, total_n = 0.0, 0
    model.eval()
    with torch.inference_mode():
        for data in dataloader:
            data = {k: v.to(device) for k, v in data.items()}
            outputs = model(**data)
            n = data["targets"].shape[0]
            total_loss += criterion(outputs, data["targets"]).item() * n
            total_n += n
            fit.update(outputs, data["targets"])
    if total_n == 0:
        raise ValueError("cannot evaluate on an empty dataset")
    return {"loss": total_loss / total_n, **fit.compute().model_dump()}
