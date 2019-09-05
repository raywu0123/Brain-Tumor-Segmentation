from math import ceil

import numpy as np

from .base import BatchSamplerBase
from utils import get_3d_from_2d, get_2d_from_3d
from ..utils import co_shuffle


class TwoDimBatchSampler(BatchSamplerBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_to_feedable(self, batch_data, batch_size, training=False, **kwargs):
        volume = batch_data['volume']
        label = batch_data['label']

        position = np.tile(np.arange(volume.shape[2]) / volume.shape[2], (len(volume),))
        two_dim_volume = get_2d_from_3d(volume)
        two_dim_label = get_2d_from_3d(label)

        if training:
            two_dim_volume, two_dim_label, position =\
                co_shuffle(two_dim_volume, two_dim_label, position)

        feedable_data_list = []
        feedable_label_list = []
        num_batch = ceil(len(two_dim_volume) / batch_size)
        for i_batch in range(num_batch):
            end_idx = min(len(two_dim_volume), (i_batch + 1) * batch_size)
            feedable_data_list.append({
                'slice': two_dim_volume[i_batch * batch_size: end_idx],
                'position': position[i_batch * batch_size: end_idx],
            })
            feedable_label_list.append(two_dim_label[i_batch * batch_size: end_idx])

        return feedable_data_list, feedable_label_list

    def reassemble(self, preds: list, test_data: dict, **kwargs):
        all_segmentations = np.concatenate(preds, axis=0)
        data_depth = test_data['volume'].shape[2]
        volume_segmentations = get_3d_from_2d(all_segmentations, data_depth)
        return volume_segmentations


class AnisotropicTwoDimBatchSampler(BatchSamplerBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_to_feedable(self, batch_data, batch_size, training=False, **kwargs):
        volume = batch_data['volume']
        label = batch_data['label']

        feedable_data_list, feedable_label_list = [], []
        if training:
            first_axes = [np.random.choice([-1, -2, -3])]
        else:
            first_axes = [-1, -2, -3]

        for axis in first_axes:
            data_list, label_list = self._convert_to_feedable_along_axis(
                volume,
                label,
                batch_size,
                axis,
                training,
            )
            feedable_data_list.extend(data_list)
            feedable_label_list.extend(label_list)
        return feedable_data_list, feedable_label_list

    @staticmethod
    def _convert_to_feedable_along_axis(volume, label, batch_size, axis, training):
        volume = np.swapaxes(volume, -3, axis)
        label = np.swapaxes(label, -3, axis)

        position = np.tile(np.arange(volume.shape[2]) / volume.shape[2], (len(volume),))
        two_dim_volume = get_2d_from_3d(volume)
        two_dim_label = get_2d_from_3d(label)

        if training:
            two_dim_volume, two_dim_label, position =\
                co_shuffle(two_dim_volume, two_dim_label, position)

        feedable_data_list, feedable_label_list = [], []
        num_batch = ceil(len(two_dim_volume) / batch_size)
        for i_batch in range(num_batch):
            end_idx = min(len(two_dim_volume), (i_batch + 1) * batch_size)
            feedable_data_list.append({
                'slice': two_dim_volume[i_batch * batch_size: end_idx],
                'position': position[i_batch * batch_size: end_idx],
            })
            feedable_label_list.append(two_dim_label[i_batch * batch_size: end_idx])
        return feedable_data_list, feedable_label_list

    def reassemble(self, preds: list, test_data: dict, **kwargs):
        start_id = 0
        batch_size, class_num = preds[0].shape[:2]
        volume_num = len(test_data['volume'])
        ensemble_pred = np.zeros([volume_num, class_num, *self.data_format])
        for axis in [-1, -2, -3]:
            num_batch = ceil(volume_num * self.data_format[axis] / batch_size)
            all_segmentations = np.concatenate(preds[start_id: start_id + num_batch], axis=0)
            volume_segmentations = get_3d_from_2d(all_segmentations, self.data_format[axis])
            volume_segmentations = np.swapaxes(volume_segmentations, -3, axis)
            ensemble_pred += volume_segmentations
            start_id += num_batch

        ensemble_pred /= 3
        return ensemble_pred
