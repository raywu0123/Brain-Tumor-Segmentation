from .base import Segmentation2DModelBase


class ToyModel(Segmentation2DModelBase):
    def __init__(
            self,
            num_units: int = 64,
            channels: int = 1,
            depth: int = 200,
            height: int = 200,
            width: int = 200,
            metadata_dim: int = 0,
    ):
        self.num_units = num_units
        self.data_channels = channels
        self.data_depth = depth
        self.data_height = height
        self.data_width = width
        self.metadata_dim = metadata_dim

    def fit(self, training_data, validation_data, **kwargs):
        # TODO
        pass

    def fit_generator(self, training_data_generator, validation_data_generator, **kwargs):
        # TODO
        pass

    def predict(self, x_test, **kwargs):
        return 0
