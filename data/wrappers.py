import multiprocessing as mp
import time

import numpy as np

from .base import DataGeneratorWrapperBase, DataGeneratorBase
from .augmentations.volume_augmentation import VolumeAugmentor


class AsyncDataGeneratorWrapper(DataGeneratorWrapperBase):

    def __init__(self, data_generator, max_q_size=10):
        super().__init__(data_generator)

        self.max_q_size = max_q_size
        self.data_queue = mp.Queue(maxsize=max_q_size)
        self.process = mp.Process(target=self._put_data_into_queue, daemon=True)
        self.process.start()

    def __call__(self, batch_size):
        if batch_size > self.max_q_size:
            raise ValueError(
                f'batch size too large, should be <= {self.max_q_size}, got {batch_size}'
            )

        while self.data_queue.qsize() < batch_size:
            time.sleep(0.1)

        data_list = [self.data_queue.get() for _ in range(batch_size)]
        batch_data = {}
        for key in data_list[0].keys():
            batch_value = np.concatenate([
                data[key] for data in data_list
            ], axis=0)
            batch_data[key] = batch_value

        return batch_data

    def _put_data_into_queue(self):
        while True:
            if not self.data_queue.full():
                data = self.data_generator(batch_size=1)
                self.data_queue.put(data)


class AugmentedDataGeneratorWrapper(DataGeneratorWrapperBase):

    def __init__(self, data_generator: DataGeneratorBase):
        super().__init__(data_generator)
        self.augmentor = VolumeAugmentor(data_generator.data_format)

    def __call__(self, batch_size):
        batch_data = self.data_generator(batch_size)
        batch_volume, batch_label = batch_data['volume'], batch_data['label']
        augmented_batch_volume, augmented_batch_label = \
            self.augmentor.co_transform(batch_volume, batch_label)
        batch_data['volume'], batch_data['label'] = augmented_batch_volume, augmented_batch_label
        return batch_data


class NormalizedDataGeneratorWrapper(DataGeneratorWrapperBase):

    def __init__(self, data_generator: DataGeneratorBase):
        super().__init__(data_generator)

    def __call__(self, batch_size):
        batch_data = self.data_generator(batch_size)
        batch_data['volume'] = self._normalize_volume(batch_data['volume'])
        return batch_data

    @staticmethod
    def _normalize_volume(batch_volume):
        if not batch_volume.ndim == 5:
            raise ValueError('input is not a volume')
        mean = np.mean(batch_volume, axis=tuple(range(2, batch_volume.ndim)), keepdims=True)
        std = np.std(batch_volume, axis=tuple(range(2, batch_volume.ndim)), keepdims=True)
        return (batch_volume - mean) / std

    def __len__(self):
        return len(self.data_generator)
