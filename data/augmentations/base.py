from abc import ABC, abstractmethod


class AugmentorBase(ABC):

    @abstractmethod
    def co_transform(self, batch_data, batch_label, **kwargs):
        pass
