from unittest import TestCase

import torch

from ..high_resolution_compact_network import CustomConv


class ModuleTests(TestCase):

    def setUp(self) -> None:
        self.in_chns = 3
        self.out_chns = self.in_chns
        self.kernel_size = 3
        self.repeats = 4

    def test_custom_conv(self):
        for dilation in [2, 4]:
            layer = CustomConv(
                self.in_chns,
                self.out_chns,
                dilation,
                self.kernel_size,
                self.repeats,
                batch_norm=False,
            )

        x = torch.ones([1, self.in_chns, 17, 18, 19])  # N, C, D, H, W
        y = layer(x)
        self.assertTrue(x.size() == y.size())
