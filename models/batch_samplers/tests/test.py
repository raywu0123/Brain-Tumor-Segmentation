from unittest import TestCase

import numpy as np

from ..two_dim import TwoDimBatchSampler
from ..patch3d import Patch3DBatchSampler


class TwoDimBatchSamplerTestCase(TestCase):

    def setUp(self):
        self.batch_volume = np.random.random([10, 9, 8, 7, 6])
        self.batch_label = np.random.randint(2, size=[10, 2, 8, 7, 6])
        self.sampler = TwoDimBatchSampler()
        self.batch_size = 2

    def test_reverse(self):
        batch_data = {'volume': self.batch_volume, 'label': self.batch_label}
        _, batch_label_list = self.sampler.convert_to_feedable(
            batch_data, batch_size=self.batch_size
        )
        reversed_batch_label = self.sampler.reassemble(batch_label_list, batch_data)
        self.assertTrue(
            np.all(reversed_batch_label == self.batch_label)
        )


class Patch3DBatchSamplerTestCase(TestCase):

    def setUp(self):
        self.batch_volume = np.random.random([5, 9, 64, 128, 200])
        self.batch_label = np.random.randint(2, size=[5, 2, 64, 128, 200])
        self.sampler = Patch3DBatchSampler()
        self.batch_size = 3

    def test_reverse(self):
        batch_data = {'volume': self.batch_volume, 'label': self.batch_label}
        _, batch_label_list = self.sampler.convert_to_feedable(
            batch_data, batch_size=self.batch_size
        )
        reversed_batch_label = self.sampler.reassemble(batch_label_list, batch_data)
        self.assertTrue(
            np.all(reversed_batch_label == self.batch_label)
        )
