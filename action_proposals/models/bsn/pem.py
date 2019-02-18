from typing import Tuple

import torch
from torch.nn import Module
import torch.nn.functional as F


class Pem(torch.nn.Module):
    def __init__(self, in_features=32):
        super(Pem, self).__init__()
        self.fc1 = torch.nn.Linear(in_features, 512)
        self.fc2 = torch.nn.Linear(512, 1)

    def forward(self, batch_features: torch.Tensor) -> torch.Tensor:
        """Proposal Evaluate Module

        :param batch_features: [B, 24]
        :return: scores. [B, 1]
        """
        return F.sigmoid(self.fc2(F.relu(self.fc1(batch_features))))
