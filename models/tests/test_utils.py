from unittest import TestCase
import torch
import numpy as np
from ..utils import MobileSelfAttention3D, SelfAttention3D


class DiceLossTestCase(TestCase):
    def setUp(self):
        self.test_pred = np.array([
            [[0.1, 0.2, 0.3],
             [0.4, 0.5, 0.6],
             [0.7, 0.8, 0.9]],
        ])
        self.test_tar = np.array([
            [[1., 0., 1.],
             [0., 1., 0.],
             [1., 1., 0.]],
        ])


class SelfAttentionTestCase(TestCase):

    def setUp(self) -> None:
        B, C, D, H, W = 2, 8, 4, 5, 6
        self.x = torch.randn(B, C, D, H, W)
        self.vanilla_layer = SelfAttention3D(in_dim=C)
        self.mobile_layer = MobileSelfAttention3D(in_dim=C)

    def test_mobile_self_attention(self):
        out = self.mobile_layer(self.x)
        self.assertTrue(out.shape == self.x.shape)

    def test_self_attention(self):
        out = self.vanilla_layer(self.x)
        self.assertTrue(out.shape == self.x.shape)
