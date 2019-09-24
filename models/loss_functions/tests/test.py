import unittest

import torch
import torch.nn.functional as F
import numpy as np

from utils import to_one_hot_label
from metrics import soft_dice as soft_dice_metric
from metrics import cross_entropy as crossentropy_metric
from ..dice import naive_soft_dice_score
from ..misc import weighted_cross_entropy


class LossFunctionTestCase(unittest.TestCase):

    def setUp(self):
        N, C, D, H, W = 1, 3, 5, 7, 9
        self.class_num = C
        self.logits = np.random.random((N, C, D, H, W))
        self.pred = F.softmax(torch.Tensor(self.logits), dim=1).data.numpy()
        self.tar_ids = np.random.choice(
            [0, 1, 2],
            size=(N, D, H, W),
            p=[0.5, 0.4, 0.1],
        )
        self.tar = to_one_hot_label(self.tar_ids, class_num=C)

    def test_naive_dice(self):
        metric_dices = []
        for p, t in zip(self.pred, self.tar):
            for class_idx in range(1, self.class_num):
                metric_dices.append(soft_dice_metric(p[class_idx], t[class_idx]))

        metric_dice = np.mean(metric_dices)
        torch_dice, _ = naive_soft_dice_score(torch.Tensor(self.logits), self.tar)
        torch_dice = torch_dice.item()
        self.assertAlmostEqual(metric_dice, torch_dice, places=2)

    def test_crossentropy(self):
        metric_crossentropy = crossentropy_metric(self.pred, self.tar_ids)
        torch_crossentropy, _ = weighted_cross_entropy(torch.Tensor(self.logits), self.tar_ids)
        torch_crossentropy = torch_crossentropy.item()
        self.assertAlmostEqual(metric_crossentropy, torch_crossentropy, places=2)
