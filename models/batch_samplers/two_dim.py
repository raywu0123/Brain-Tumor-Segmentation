from math import ceil

import numpy as np

from .base import BatchSamplerBase


class TwoDimBatchSampler(BatchSamplerBase):

    def convert_to_feedable(self, batch_data, batch_size):
        volume = batch_data['volume']
        label = batch_data['label']
        two_dim_volume = self._get_2d_from_3d(volume)
        two_dim_label = self._get_2d_from_3d(label)

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
        volume_segmentations = self._get_3d_from_2d(all_segmentations, data_depth)
        return volume_segmentations

    @staticmethod
    def _get_2d_from_3d(batch_volume):
        assert (batch_volume.ndim == 5)
        batch_volume = np.transpose(batch_volume, (0, 2, 1, 3, 4))
        batch_image = batch_volume.reshape(-1, *batch_volume.shape[-3:])
        return batch_image

    @staticmethod
    def _get_3d_from_2d(batch_image, data_depth):
        assert (batch_image.ndim == 4)
        batch_volume = batch_image.reshape(-1, data_depth, *batch_image.shape[-3:])
        batch_volume = batch_volume.transpose([0, 2, 1, 3, 4])
        return batch_volume
