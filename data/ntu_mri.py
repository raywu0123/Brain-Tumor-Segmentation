from .base import DataInterface


class NTUMRI(DataInterface):
    def __init__(self):
        self.description = 'NTUMRI'

    def get_training_data(self):
        # TODO
        pass

    def get_data_format(self):
        data_format = {
            "channels": 1,
            "depth": 160,
            "height": 217,
            "width": 217,
        }
        return data_format
