from typing import Tuple

import torch
from torch.nn import Module
import torch.nn.functional as F


class Pem(Module):
    def __init__(self, in_features=32):
        super(Pem, self).__init__()
        # self.fc1 = torch.nn.Linear(in_features, 512)
        # self.fc1.weight = 0.1 * self.fc1.weight
        #
        # self.fc2 = torch.nn.Linear(512, 1)
        # self.fc2.weight = 0.1 * self.fc1.weight
        self.w1 = torch.nn.Parameter(torch.randn(32, 256))
        self.w2 = torch.nn.Parameter(torch.randn(256, 1))
        self.bias1 = torch.nn.Parameter(torch.randn(256))
        self.bias2 = torch.nn.Parameter(torch.randn(1))
        self.register_parameter("W1", self.w1)
        self.register_parameter("W2", self.w2)
        self.register_parameter("B1", self.bias1)
        self.register_parameter("B2", self.bias2)

    def forward(self, batch_features: torch.Tensor) -> torch.Tensor:
        """Proposal Evaluate Module

        :param batch_features: [P, 32]
        :return: scores. [P]
        """
        # scores = torch.sigmoid(self.fc2(F.relu(self.fc1(batch_features))))
        out1 = F.relu(0.1 * torch.matmul(batch_features, self.w1) + self.bias1)
        scores = torch.sigmoid(0.1 * torch.matmul(out1, self.w2) + self.bias2)
        return torch.squeeze(scores)


class PemLoss(Module):
    def __init__(self):
        super(PemLoss, self).__init__()
        self.sl1_loss = torch.nn.SmoothL1Loss()

    def forward(self, pred_scores: torch.Tensor, gt_iou: torch.Tensor):
        """Sample from a batch, then calc smooth l1 loss.

        :param pred_scores: [P]
        :param gt_iou: [P]
        :return: [1] the smooth l1 loss.
        """
        # pred_scores = batch_pred_scores.reshape(-1)
        # gt_iou = batch_gt_iou.view(-1)

        # get threshold mask
        hmask = (gt_iou > 0.6).type_as(gt_iou)
        mmask = ((gt_iou <= 0.6) & (gt_iou > 0.2)).type_as(gt_iou)
        lmask = (gt_iou <= 0.2).type_as(gt_iou)

        num_h = hmask.sum()
        num_m = mmask.sum()
        num_l = lmask.sum()

        # random sample from m and l
        ratio_m = 1 * num_h / num_m
        if ratio_m > 1:
            ratio_m = 1
        smmask = torch.rand(mmask.shape).type_as(gt_iou)
        smmask = smmask * mmask
        smmask = (smmask > (1 - ratio_m)).type_as(gt_iou)

        ratio_l = 2 * num_h / num_l
        if ratio_l > 1:
            ratio_l = 1
        slmask = torch.rand(lmask.shape).type_as(gt_iou)
        slmask = slmask * lmask
        slmask = (slmask > (1 - ratio_l)).type_as(gt_iou)

        iou_weights = hmask + smmask + slmask
        pred_scores = pred_scores * iou_weights
        gt_iou = gt_iou * iou_weights

        return self.sl1_loss(pred_scores, gt_iou)
