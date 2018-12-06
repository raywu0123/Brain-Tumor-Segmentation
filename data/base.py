from abc import ABC, abstractmethod


class DataGeneratorBase(ABC):

    @abstractmethod
    def __call__(self, batch_size: int) -> dict:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @property
    def data_format(self):
        return self._data_format
