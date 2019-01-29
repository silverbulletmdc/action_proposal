from typing import Tuple

import torch
from torch.nn import Module
import torch.nn.functional as F


class Tem(Module):
    """Temporal evalutate module

    Shape:
        - X: :math:`(N, in_features, L)`
        - Output: :math:`(N, 3, L)`

    Args:
        input_features (int): input features length
    """

    def __init__(self, input_features):

        super(Tem, self).__init__()
        self.conv1 = torch.nn.Conv1d(input_features, 512, 3, 1, 1)
        self.conv2 = torch.nn.Conv1d(512, 512, 3, 1, 1)
        self.conv3 = torch.nn.Conv1d(512, 3, 3, 1, 1)

    def forward(self, x):
        """
        Args:
            x:
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        output = torch.sigmoid(0.1 * self.conv3(x))

        return output


class TemLoss(Module):
    r"""

    """
    def __init__(self):
        super(TemLoss, self).__init__()

    def forward(self, pred_anchors, gt_scores):
        loss_start, num_sample_start, ratio_start = self._binary_logistic_loss(gt_scores[:, 0, :], pred_anchors[:, 0, :])
        loss_action, num_sample_action, ratio_action = self._binary_logistic_loss(gt_scores[:, 1, :], pred_anchors[:, 1, :])
        loss_end, num_sample_end, ratio_end = self._binary_logistic_loss(gt_scores[:, 2, :], pred_anchors[:, 2, :])

        return loss_start, loss_action, loss_end


    @staticmethod
    def _binary_logistic_loss(gt_scores: torch.Tensor, pred_anchors: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        Calc binary logistic loss.

        :param gt_scores:
        :param pred_anchors:
        :return: loss, num_positive, ratio

        Shape:
            - gt_scores: :math:`(N, L)`
            - pred_anchors: :math:`(N, L)`
            - loss: :math:`(N, 1)`
            - num_positive: :math:`(N, 1)`
            - ratio: :math:`(N, 1)`, where :math:`ratio=num_entries/num_positive`
        """
        pmask: torch.Tensor = (gt_scores >= 0.5).type(gt_scores.dtype)
        num_positive = pmask.sum()
        num_entries = pmask.shape[0] * pmask.shape[1]
        ratio = num_entries / num_positive
        coef_0: torch.Tensor = 0.5 * ratio / (ratio - 1)
        coef_1 = coef_0 * (ratio - 1)
        # coef_0 = coef_0.unsqueeze(1).expand(*pmask.shape)
        # coef_1 = coef_1.unsqueeze(1).expand(*pmask.shape)
        loss = coef_1 * pmask * pred_anchors.log() + coef_0 * (1 - pmask) * torch.log(1.0 - pred_anchors)
        loss = -torch.mean(loss)

        return loss, num_positive, ratio
