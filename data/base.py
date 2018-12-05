from abc import ABC, abstractmethod

from utils import MetricClass


class DataGeneratorFactoryBase(ABC):

    _metric = MetricClass

    @abstractmethod
    def get_training_data_generator(self, random=True):
        pass

    @abstractmethod
    def get_testing_data_generator(self, random=True):
        pass

    @property
    def metric(self):
        return self._metric

    @property
    @abstractmethod
    def data_format(self) -> dict:
        pass


class DataGeneratorBase(ABC):

    @abstractmethod
    def __call__(self, batch_size: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass
