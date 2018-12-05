from abc import ABC, abstractmethod


class TrainerBase(ABC):

    @abstractmethod
    def fit_generator(self, training_data_generator, validation_data_generator, metric, **kwargs):
        pass

    @abstractmethod
    def save(self):
        pass

    @abstractmethod
    def load(self, checkpoint_path):
        pass
