from .base import Segmentation2DModelBase


class ToyModel(Segmentation2DModelBase):
    def __init__(
            self,
            **kwargs,
    ):
        self.num_units = kwargs['num_units']
        self.data_channels = kwargs['channels']
        self.data_depth = kwargs['depth']
        self.data_height = kwargs['height']
        self.data_width = kwargs['width']

    def fit(self, training_data, validation_data, **kwargs):
        # TODO
        pass

    def predict(self, x_test, **kwargs):
        return 0
