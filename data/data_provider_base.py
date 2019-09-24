from abc import ABC, abstractmethod

from metrics import ClasswiseMetric
from .wrappers import (
    AsyncDataGeneratorWrapper,
    NormalizedDataGeneratorWrapper,
    AugmentedDataGeneratorWrapper,
)


class DataProviderBase(ABC):

    _metric = ClasswiseMetric

    def get_full_data_generator(self, **kwargs):
        return self._get_data_generator(
            self.train_ids + self.test_ids,
            **{**kwargs, 'augmentation': False},
        )

    def get_testing_data_generator(self, **kwargs):
        return self._get_data_generator(
            self.test_ids,
            **{**kwargs, 'augmentation': False, 'random': False},
        )

    def get_training_data_generator(self, **kwargs):
        return self._get_data_generator(self.train_ids, **kwargs)

    def _get_data_generator(self, data_ids, augmentation=False, async_load=False, **kwargs):
        data_generator = self._get_raw_data_generator(data_ids, **kwargs)
        data_generator = NormalizedDataGeneratorWrapper(data_generator)
        if augmentation:
            data_generator = AugmentedDataGeneratorWrapper(data_generator)
        if async_load:
            data_generator = AsyncDataGeneratorWrapper(data_generator)
        return data_generator

    @abstractmethod
    def _get_raw_data_generator(self, data_ids, **kwargs):
        pass

    @property
    def metric(self):
        return self._metric

    @property
    @abstractmethod
    def data_format(self) -> dict:
        pass

    def __len__(self):
        return len(self.train_ids)
