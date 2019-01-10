from unittest import TestCase

import numpy as np

from ..two_dim import TwoDimBatchSampler
from ..uniform_patch3d import UniformPatch3DBatchSampler
from ..center_patch3d import CenterPatch3DBatchSampler


class TwoDimBatchSamplerTestCase(TestCase):

    def setUp(self):
        self.batch_volume = np.random.random([10, 9, 8, 7, 6])
        self.batch_label = np.random.randint(2, size=[10, 2, 8, 7, 6])
        self.sampler = TwoDimBatchSampler(
            data_format={
                'channels': 9,
                'depth': 8,
                'height': 7,
                'width': 6,
                'class_num': 2,
            }
        )
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
        self.data_format = {
            'channels': 5,
            'depth': 64,
            'height': 128,
            'width': 200,
            'class_num': 2,
        }
        self.n_data = 5
        self.batch_volume = np.random.random([
            self.n_data,
            self.data_format['channels'],
            self.data_format['depth'],
            self.data_format['height'],
            self.data_format['width']]
        )
        self.batch_label = np.random.choice(
            [0, 1],
            size=[
                self.n_data,
                self.data_format['class_num'],
                self.data_format['depth'],
                self.data_format['height'],
                self.data_format['width'],
            ],
            p=[0.01, 0.99],
        )
        self.batch_data = {'volume': self.batch_volume, 'label': self.batch_label}

        self.uniform_sampler = UniformPatch3DBatchSampler(data_format=self.data_format)
        self.center_sampler = CenterPatch3DBatchSampler(data_format=self.data_format)

        self.batch_size = 3

    def test_reverse(self):
        _, batch_label_list = self.uniform_sampler.convert_to_feedable(
            self.batch_data, batch_size=self.batch_size, training=False
        )
        reversed_batch_label = self.uniform_sampler.reassemble(batch_label_list, self.batch_data)
        self.assertTrue(
            np.all(reversed_batch_label == self.batch_label)
        )

    def test_center_sampler(self):
            _, batch_label_list = self.uniform_sampler.convert_to_feedable(
                self.batch_data, batch_size=self.batch_size, training=True
            )
            _, batch_label_list = self.center_sampler.convert_to_feedable(
                self.batch_data, batch_size=self.batch_size, training=True
            )
