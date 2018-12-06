from unittest import TestCase

import numpy as np


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
