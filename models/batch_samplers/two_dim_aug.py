from math import ceil

import numpy as np

from .base import BatchSamplerBase
from utils import get_3d_from_2d, get_2d_from_3d
from ..utils import co_shuffle


class TwoDimAugBatchSampler(BatchSamplerBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_to_feedable(self, batch_data, batch_size, training=False, **kwargs):
        volume = batch_data['volume']
        label = batch_data['label']

        feedable_data_list, feedable_label_list = [], []
        for orientation in (-1, -2, -3):
            data_list, label_list = self.prepare_feedable_list(
                volume,
                label,
                orientation,
                training,
                batch_size,
            )
            feedable_data_list.extend(data_list)
            feedable_label_list.extend(label_list)
        return feedable_data_list, feedable_label_list

    @staticmethod
    def prepare_feedable_list(volume, label, orientation, training, batch_size):
        volume = volume.swapaxes(-3, orientation)
        label = label.swapaxes(-3, orientation)
        two_dim_volume = get_2d_from_3d(volume)
        two_dim_label = get_2d_from_3d(label)
        if training:
            two_dim_volume, two_dim_label = co_shuffle(two_dim_volume, two_dim_label)

        data_list, label_list = [], []
        num_batch = ceil(len(two_dim_volume) / batch_size)
        for i_batch in range(num_batch):
            end_idx = min(len(two_dim_volume), (i_batch + 1) * batch_size)
            data_list.append({
                'slice': two_dim_volume[i_batch * batch_size: end_idx],
            })
            label_list.append(two_dim_label[i_batch * batch_size: end_idx])
        return data_list, label_list

    def reassemble(self, preds: list, test_data: dict, **kwargs):
        batch_size = len(preds[0])
        volume = test_data['volume']
        pred_start_idx = 0
        predictions = []
        for orientation in (-1, -2, -3):
            slicing_axes_size = volume.shape[orientation]
            pred_size = ceil(len(volume) * slicing_axes_size / batch_size)
            all_segmentations = np.concatenate(
                preds[pred_start_idx: pred_start_idx + pred_size],
                axis=0,
            )
            volume_segmentations = get_3d_from_2d(all_segmentations, slicing_axes_size)
            volume_segmentations = volume_segmentations.swapaxes(-3, orientation)
            predictions.append(volume_segmentations)
            pred_start_idx += pred_size
        return np.mean(predictions, axis=0)
