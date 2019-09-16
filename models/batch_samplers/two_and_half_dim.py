from math import ceil

import numpy as np

from .base import BatchSamplerBase
from ..utils import co_shuffle


class TwoAndHalfDimBatchSampler(BatchSamplerBase):

    def __init__(self, depth: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.depth = depth

    def convert_to_feedable(self, batch_data, batch_size, training=False, **kwargs):
        volume = batch_data['volume']
        label = batch_data['label']

        position = np.tile(np.linspace(0., 1., volume.shape[-3]), (len(volume), 1))  # shape: (N, D)
        new_depth = ceil(volume.shape[-3] / self.depth) * self.depth
        volume = np.resize(
            volume, [len(volume), volume.shape[1], new_depth, volume.shape[-2], volume.shape[-1]]
        )
        label = np.resize(
            label, [len(label), new_depth, label.shape[-2], label.shape[-1]]
        )
        position = np.resize(
            position, [len(position), new_depth]
        )
        volume = volume.reshape([
            len(volume) * new_depth // self.depth,
            volume.shape[1] * self.depth,
            volume.shape[-2],
            volume.shape[-1],
        ])
        label = label.reshape([
            len(label) * new_depth // self.depth,
            self.depth,
            label.shape[-2],
            label.shape[-1],
        ])
        position = position.reshape(
            len(position) * new_depth // self.depth,
            self.depth,
        )
        position = np.mean(position, axis=-1)

        if training:
            volume, label, position = co_shuffle(volume, label, position)

        feedable_data_list = []
        feedable_label_list = []
        num_batch = ceil(len(volume) / batch_size)
        for i_batch in range(num_batch):
            feedable_data_list.append({
                'slice': volume[i_batch * batch_size: (i_batch + 1) * batch_size],
                'position': position[i_batch * batch_size: (i_batch + 1) * batch_size],
            })
            feedable_label_list.append(label[i_batch * batch_size: (i_batch + 1) * batch_size])

        return feedable_data_list, feedable_label_list

    def reassemble(self, preds: list, test_data: dict, **kwargs):
        seg = np.concatenate(preds, axis=0)  # shape: (N', class_num, self.depth, H, W)
        class_num = seg.shape[1]
        seg = seg.transpose([0, 2, 3, 4, 1])
        test_volume = test_data['volume']
        data_depth, data_height, data_width = test_volume.shape[2:]
        seg = seg.reshape([
            len(test_volume),
            -1,
            data_height,
            data_width,
            class_num,
        ])
        seg = np.resize(
            seg, [len(test_volume), data_depth, data_height, data_width, class_num]
        )
        seg = seg.transpose([0, 4, 1, 2, 3])
        return seg

    def get_data_format(self, input_data_format):
        return {
            **input_data_format,
            'channels': input_data_format['channels'] * self.depth,
        }
