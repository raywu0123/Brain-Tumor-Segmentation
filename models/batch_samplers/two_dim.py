from math import ceil

import numpy as np

from .base import BatchSamplerBase
from utils import get_3d_from_2d, get_2d_from_3d
from ..utils import co_shuffle


class TwoDimBatchSampler(BatchSamplerBase):

    def convert_to_feedable(self, batch_data, batch_size, training=False, **kwargs):
        volume = batch_data['volume']
        label = batch_data['label']
        two_dim_volume = get_2d_from_3d(volume)
        two_dim_label = get_2d_from_3d(label)
        if training:
            two_dim_volume, two_dim_label = co_shuffle(two_dim_volume, two_dim_label)

        feedable_data_list = []
        feedable_label_list = []
        num_batch = ceil(len(two_dim_volume) / batch_size)
        for i_batch in range(num_batch):
            end_idx = min(len(two_dim_volume), (i_batch + 1) * batch_size)
            feedable_data_list.append(two_dim_volume[i_batch * batch_size: end_idx])
            feedable_label_list.append(two_dim_label[i_batch * batch_size: end_idx])

        return feedable_data_list, feedable_label_list

    def reassemble(self, preds: list, test_data: dict, **kwargs):
        all_segmentations = np.asarray([pred for pred in preds])
        all_segmentations = all_segmentations.reshape(-1, *all_segmentations.shape[2:])
        data_depth = test_data['volume'].shape[2]
        volume_segmentations = get_3d_from_2d(all_segmentations, data_depth)
        return volume_segmentations
