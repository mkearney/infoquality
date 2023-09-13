import torch
from torch.utils.data import Dataset


class InputsDataset(Dataset):
    """
    Dataset for the messages.

    ### Attributes
        - `input_ids`: tensor of message indices
        - `attention_mask`: tensor of attention masks
        - `targets`: tensor of target indices
    """

    def __init__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        targets: torch.Tensor,
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "targets": self.targets[idx],
        }
