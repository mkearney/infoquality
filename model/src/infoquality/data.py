from typing import Dict, List

import torch
from infoquality.preprocessor import Preprocessor
from torch.utils.data import Dataset


class MessagesDataset(Dataset):
    def __init__(
        self,
        messages: List[str],
        labels: List[str],
        label_map: Dict[str, int],
        preprocessor: Preprocessor,
    ):
        self.messages = messages
        self.indices: List[torch.Tensor] = preprocessor(messages)
        self.labels = labels
        self.targets = [label_map[label] for label in self.labels]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.indices[idx], self.targets[idx]
