from math import ceil

import numpy as np

from .base import BatchSamplerBase
from preprocess_tools.image_utils import crop_or_pad_to_shape
from .utils import flatten


class UniformPatch3DBatchSampler(BatchSamplerBase):

    patch_size = np.array((152, 128, 128))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_to_feedable(self, batch_data, batch_size, training=False, **kwargs):
        batch_volume = batch_data['volume']
        batch_label = batch_data['label']

        feedable_data_list = []
        feedable_label_list = []

        index_lists = self._generate_index_lists(batch_volume.shape, random=training)
        batch_index_lists = np.array_split(index_lists, ceil(len(index_lists) / batch_size))

        for batch_indexes in batch_index_lists:
            batch_patch_volume, batch_patch_label = \
                self._sample_by_batch_lists(batch_volume, batch_label, batch_indexes)
            feedable_data_list.append(batch_patch_volume)
            feedable_label_list.append(batch_patch_label)

        return feedable_data_list, feedable_label_list

    def _sample_by_batch_lists(self, batch_volume, batch_label, batch_indexes):
        batch_patch_volume = np.zeros([len(batch_indexes), batch_volume.shape[1], *self.patch_size])
        batch_patch_label = np.zeros([len(batch_indexes), *self.patch_size], dtype=np.uint8)

        for idx, index in enumerate(batch_indexes):
            batch_patch_volume[idx] = self._sample_by_index(batch_volume, index)
            batch_patch_label[idx] = self._sample_by_index(batch_label, index)

        return batch_patch_volume, batch_patch_label

    def _sample_by_index(self, batch_data, index_list):
        if batch_data.ndim == 5:
            patch = batch_data[
                index_list[0], :,
                index_list[1]: index_list[1] + self.patch_size[0],
                index_list[2]: index_list[2] + self.patch_size[1],
                index_list[3]: index_list[3] + self.patch_size[2],
            ]
            patch = crop_or_pad_to_shape(patch, [patch.shape[0], *self.patch_size])
        elif batch_data.ndim == 4:
            patch = batch_data[
                index_list[0],
                index_list[1]: index_list[1] + self.patch_size[0],
                index_list[2]: index_list[2] + self.patch_size[1],
                index_list[3]: index_list[3] + self.patch_size[2],
            ]
            patch = crop_or_pad_to_shape(patch, self.patch_size.tolist())
        else:
            raise ValueError(f'Invalid shape {batch_data.shape}')
        return patch

    def _generate_index_lists(self, volume_shape, random):
        volume_shape = np.asarray(volume_shape)
        if random:
            num_patches = volume_shape[0] \
                * ceil(volume_shape[1] / self.patch_size[0]) \
                * ceil(volume_shape[2] / self.patch_size[1]) \
                * ceil(volume_shape[3] / self.patch_size[2])

            random_range = np.array([volume_shape[0], *(volume_shape[2:] - self.patch_size)])
            lists = np.floor(np.random.rand(num_patches, 4) * random_range).astype(int)
            return lists
        else:
            lists = []
            for n in range(volume_shape[0]):
                for d in range(ceil(volume_shape[2] / self.patch_size[0])):
                    for h in range(ceil(volume_shape[3] / self.patch_size[1])):
                        for w in range(ceil(volume_shape[4] / self.patch_size[2])):
                            lists.append([
                                n,
                                d * self.patch_size[0],
                                h * self.patch_size[1],
                                w * self.patch_size[2],
                            ])
        return lists

    def reassemble(self, prediction: list, test_data: dict, **kwargs):
        test_volume = test_data['volume']
        reassembled_prediction = np.zeros([
            test_volume.shape[0],
            prediction[0].shape[1],
            *test_volume.shape[2:],
        ])
        index_lists = self._generate_index_lists(test_volume.shape, random=False)
        prediction = flatten(prediction)

        for pred_patch, index_list in zip(prediction, index_lists):
            target_shape = test_volume[
                0, 0,
                index_list[1]: index_list[1] + self.patch_size[0],
                index_list[2]: index_list[2] + self.patch_size[1],
                index_list[3]: index_list[3] + self.patch_size[2],
            ].shape
            cropped_pred_patch = crop_or_pad_to_shape(
                pred_patch, [pred_patch.shape[0], *target_shape]
            )
            reassembled_prediction[
                index_list[0], :,
                index_list[1]: index_list[1] + self.patch_size[0],
                index_list[2]: index_list[2] + self.patch_size[1],
                index_list[3]: index_list[3] + self.patch_size[2],
            ] = cropped_pred_patch
        return reassembled_prediction
