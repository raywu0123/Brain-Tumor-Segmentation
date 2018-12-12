from unittest import TestCase

import numpy as np

from utils import (
    hard_max,
    get_2d_from_3d,
    get_3d_from_2d,
)


class MetricTestCase(TestCase):

    def setUp(self):
        self.x = np.array(
            [[
                [0.1, 0.5, 0.7],
                [0.9, 0.5, 0.3],
            ]]
        )

    def test_hard_max(self):
        hard_max_x = hard_max(self.x)
        self.assertEqual(
            np.all(
                hard_max_x == np.array(
                    [[
                        [0., 1., 1.],
                        [1., 0., 0.],
                    ]]
                )
            ),
            True,
        )


class DimensionConvertTestCase(TestCase):

    def setUp(self):
        self.x = np.random.random([10, 3, 5, 20, 20])
        self.two_d_x = get_2d_from_3d(self.x)
        self.reverse_x = get_3d_from_2d(self.two_d_x, self.x.shape[2])

    def test_reverse(self):
        self.assertEqual(
            np.all(self.x == self.reverse_x),
            True,
        )

    def test_two_d_shape(self):
        self.assertEqual(
            np.all(self.two_d_x.shape == np.array([50, 3, 20, 20])),
            True,
        )

    def test_reverse_shape(self):
        self.assertEqual(
            np.all(self.reverse_x.shape == self.x.shape),
            True,
        )
