from unittest import TestCase

import numpy as np

from ..two_dim import TwoDimBatchSampler
from ..uniform_patch3d import UniformPatch3DBatchSampler
from ..center_patch3d import CenterPatch3DBatchSampler


class TwoDimBatchSamplerTestCase(TestCase):

    def setUp(self):
        self.batch_volume = np.random.random([10, 9, 8, 7, 6])
        self.batch_label = np.random.randint(2, size=[10, 8, 7, 6])
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
            'height': 18,
            'width': 20,
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

    def test_uniform_reverse(self):
        for training in [False, True]:
            _, batch_label_list = self.uniform_sampler.convert_to_feedable(
                self.batch_data, batch_size=self.batch_size, training=training
            )

            if not training:
                mock_batch_pred_list = [label[:, np.newaxis] for label in batch_label_list]
                reversed_batch_label = self.uniform_sampler.reassemble(
                    mock_batch_pred_list,
                    self.batch_data,
                )
                self.assertTrue(
                    np.all(reversed_batch_label[:, 0] == self.batch_label)
                )

    def test_center_reverse(self):
        for training in [False, True]:
            _, batch_label_list = self.center_sampler.convert_to_feedable(
                self.batch_data, batch_size=self.batch_size, training=training
            )
            for batch in batch_label_list:
                self.assertTrue(np.all(batch.shape[-3:] == self.center_sampler.patch_size))

            if not training:
                mock_batch_pred_list = [label[:, np.newaxis] for label in batch_label_list]
                reversed_batch_label = self.center_sampler.reassemble(
                    mock_batch_pred_list,
                    self.batch_data,
                )
                self.assertTrue(
                    np.all(reversed_batch_label[:, 0] == self.batch_label)
                )
