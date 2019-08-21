from unittest import TestCase

import numpy as np

from metrics import hard_max


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
