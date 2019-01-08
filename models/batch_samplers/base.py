from abc import ABC, abstractmethod
from typing import Tuple


class BatchSamplerBase(ABC):

    def __init__(self, data_format, **kwargs):
        self.data_format = [data_format['depth'], data_format['height'], data_format['width']]

    @abstractmethod
    def convert_to_feedable(self, batch_data, batch_size, training, **kwargs) -> Tuple[list, list]:
        pass

    @abstractmethod
    def reassemble(self, prediction: list, test_data: dict, **kwargs):
        pass
