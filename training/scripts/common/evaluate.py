import torch
from common.metrics import Fit


def evaluate(
    model,
    dataloader,
    criterion,
    train_loss: float,
):
    """
    Epoch evaluation for model training

    ### Args
        - `model`: model
        - `dataloader`: dataloader
        - `criterion`: loss function
        - `train_loss`: training loss

    ### Returns
        - `metrics`: dict of metrics
    """
    losses, accs, f1s, prs, rcs = [], [], [], [], []
    fit = Fit(num_classes=model.hyperparameters.num_classes)
    model.eval()  # type: ignore
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            outputs = model(**data)  # type: ignore
            loss = criterion(outputs, data["targets"].long())
            losses.append(loss.item())
            fit_metrics = fit(outputs, data["targets"])
            accs.append(fit_metrics.acc)
            f1s.append(fit_metrics.f1)
            prs.append(fit_metrics.pr)
            rcs.append(fit_metrics.rc)
            if i == model.hyperparameters.num_steps:
                break
        # calculate means
        denom = len(losses)
        val_loss = sum(losses) / denom
        val_acc = sum(accs) / denom
        val_f1 = sum(f1s) / denom
        val_pr = sum(prs) / denom
        val_rc = sum(rcs) / denom
        metrics = {
            "loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "val_pr": val_pr,
            "val_rc": val_rc,
        }
        return metrics
