from abc import ABC, abstractmethod

import numpy as np


class DataGeneratorBase(ABC):

    def __init__(self, data_ids, data_format, random=True):
        self.data_ids = data_ids
        self._data_format = data_format
        self.random = random
        self.current_index = 0

    def __call__(self, batch_size: int) -> dict:
        if self.random:
            selected_data_ids = np.random.choice(self.data_ids, batch_size)
        else:
            selected_data_ids = self.data_ids[self.current_index: self.current_index + batch_size]
            self.current_index += batch_size
            self.current_index %= self.__len__()
        return self._get_data(selected_data_ids)

    def __len__(self) -> int:
        return len(self.data_ids)

    @abstractmethod
    def _get_data(self, data_ids):
        pass

    @property
    def data_format(self):
        return self._data_format
