from abc import ABC, abstractmethod
from typing import Tuple


class BatchSamplerBase(ABC):

    @abstractmethod
    def convert_to_feedable(self, batch_data, batch_size) -> Tuple[list, list]:
        pass

    @abstractmethod
    def reassemble(self, prediction: list, test_data: list, **kwargs):
        pass
