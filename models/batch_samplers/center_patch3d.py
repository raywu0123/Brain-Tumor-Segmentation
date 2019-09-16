from math import ceil
import random

import numpy as np

from .base import BatchSamplerBase
from .utils import flatten


class CenterPatch3DBatchSampler(BatchSamplerBase):

    patch_size = np.array((32, 128, 128), dtype=int)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_to_feedable(self, batch_data, batch_size, training=False, **kwargs):
        batch_volume = batch_data['volume']
        batch_label = batch_data['label']

        feedable_data_list = []
        feedable_label_list = []

        index_list = self._generate_index_list(batch_label, random=training)
        if len(index_list) != 0:
            index_lists = np.array_split(index_list, ceil(len(index_list) / batch_size))
        else:
            index_lists = []

        for batch_indexes in index_lists:
            batch_patch_volume, batch_patch_label = \
                self._sample_by_batch_lists(batch_volume, batch_label, batch_indexes)
            feedable_data_list.append(batch_patch_volume)
            feedable_label_list.append(batch_patch_label)

        return feedable_data_list, feedable_label_list

    def _sample_by_batch_lists(self, batch_volume, batch_label, batch_indexes):
        batch_patch_volume = np.zeros([len(batch_indexes), batch_volume.shape[1], *self.patch_size])
        batch_patch_label = np.zeros([len(batch_indexes), *self.patch_size], dtype=np.uint8)

        for idx, index in enumerate(batch_indexes):
            patch_volume = self._sample_by_index(batch_volume, index)
            patch_label = self._sample_by_index(batch_label, index)
            batch_patch_volume[
                idx, :,
                :patch_volume.shape[1],
                :patch_volume.shape[2],
                :patch_volume.shape[3],
            ] = patch_volume[:, :self.patch_size[0], :self.patch_size[1], :self.patch_size[2]]
            batch_patch_label[
                idx,
                :patch_label.shape[0],
                :patch_label.shape[1],
                :patch_label.shape[2],
            ] = patch_label[:self.patch_size[0], :self.patch_size[1], :self.patch_size[2]]

        return batch_patch_volume, batch_patch_label

    def _sample_by_index(self, batch_data, index_list):
        if batch_data.ndim == 5:
            patch = batch_data[
                index_list[0], :,
                index_list[1]: index_list[1] + self.patch_size[0],
                index_list[2]: index_list[2] + self.patch_size[1],
                index_list[3]: index_list[3] + self.patch_size[2],
            ]
        elif batch_data.ndim == 4:
            patch = batch_data[
                index_list[0],
                index_list[1]: index_list[1] + self.patch_size[0],
                index_list[2]: index_list[2] + self.patch_size[1],
                index_list[3]: index_list[3] + self.patch_size[2],
            ]
        else:
            raise ValueError(f'Invalid shape {batch_data.shape}')
        return patch

    def _generate_index_list(self, batch_label, random):
        volume_shape = batch_label.shape
        if random:
            num_patches = volume_shape[0] \
                * ceil(volume_shape[1] / self.patch_size[0]) \
                * ceil(volume_shape[2] / self.patch_size[1]) \
                * ceil(volume_shape[3] / self.patch_size[2])
            return self._generate_index_list_around_label(batch_label, num_patches)
        else:
            lists = []
            for n in range(volume_shape[0]):
                for d in range(ceil(volume_shape[-3] / self.patch_size[-3])):
                    for h in range(ceil(volume_shape[-2] / self.patch_size[-2])):
                        for w in range(ceil(volume_shape[-1] / self.patch_size[-2])):
                            lists.append([
                                n,
                                d * self.patch_size[0],
                                h * self.patch_size[1],
                                w * self.patch_size[2],
                            ])
            return lists

    def _generate_index_list_around_label(self, batch_label, num_patches):
        lists = []
        random_n_list = np.random.randint(0, len(batch_label), size=num_patches)
        indexes_dict = {}
        for random_n in random_n_list:
            if random_n not in indexes_dict:
                selected_label = batch_label[random_n]

                # find label except background
                selected_label = (selected_label != 0)
                indexes = np.stack(np.where(selected_label), axis=-1)
                indexes_dict[random_n] = indexes
            else:
                indexes = indexes_dict[random_n]

            if len(indexes) != 0:
                selected_label_position = random.choice(indexes)
                patch_position = self._get_patch_position(selected_label_position)
                selected_label_position = [random_n, *patch_position]
                lists.append(selected_label_position)
        return lists

    def _get_patch_position(self, selected_label_position):
        pos = []
        for patch_l, label_pos, data_l in zip(
                self.patch_size, selected_label_position, self.data_format
        ):
            a_max = max(data_l - patch_l, 0)
            x = label_pos - np.random.randint(0, patch_l)
            x = np.clip(x, a_min=0, a_max=a_max)
            pos.append(x)
        return pos

    def reassemble(self, prediction: list, test_data: dict, **kwargs):
        test_volume = test_data['volume']
        reassembled_prediction = np.zeros([
            test_volume.shape[0],
            prediction[0].shape[1],
            *test_volume.shape[2:],
        ])
        index_lists = self._generate_index_list(test_volume, random=False)
        prediction = flatten(prediction)

        for pred_patch, index_list in zip(prediction, index_lists):
            reassemble_patch = reassembled_prediction[
                index_list[0], :,
                index_list[1]: index_list[1] + self.patch_size[0],
                index_list[2]: index_list[2] + self.patch_size[1],
                index_list[3]: index_list[3] + self.patch_size[2]
            ]
            reassemble_patch[:, :, :, :] = pred_patch[
                :,
                :reassemble_patch.shape[-3],
                :reassemble_patch.shape[-2],
                :reassemble_patch.shape[-1],
            ]
        return reassembled_prediction
