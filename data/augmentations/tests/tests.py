from unittest import TestCase
import numpy as np

from ..volume_augmentation import VolumeAugmentor


class AugmentationTestCase(TestCase):

    def setUp(self):
        data_format = {
            'channels': 1,
            'depth': 200,
            'height': 199,
            'width': 198,
        }
        data_shape = [
            1,
            data_format['channels'],
            data_format['depth'],
            data_format['height'],
            data_format['width'],
        ]
        label_shape = [
            1,
            data_format['depth'],
            data_format['height'],
            data_format['width'],
        ]
        self.volume = np.random.random(data_shape)
        self.label = np.random.random(label_shape)
        self.identity_augmentor = VolumeAugmentor(data_format=data_format, mode='identity')
        self.nontrivial_augmentor = VolumeAugmentor(data_format=data_format)

    def test_identity_transform(self):
        transformed_volume, transformed_label = self.identity_augmentor.co_transform(
            volume=self.volume, label=self.label
        )
        self.assertEqual(np.all(transformed_volume.shape == self.volume.shape), True)
        self.assertEqual(np.all(transformed_label.shape == self.label.shape), True)

        self.assertAlmostEqual(np.sum(transformed_volume - self.volume), 0.)

    def test_nontrivial_transform(self):
        for _ in range(10):
            transformed_volume, transformed_label = self.nontrivial_augmentor.co_transform(
                volume=self.volume, label=self.label
            )
            self.assertEqual(np.all(transformed_volume.shape == self.volume.shape), True)
            self.assertEqual(np.all(transformed_label.shape == self.label.shape), True)
