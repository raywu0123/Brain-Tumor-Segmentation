class DataInterface:

    _description = None

    def get_training_data(self):
        raise NotImplementedError(f'{self.__class__.__name__} does not implement get_training_data')

    @property
    def description(self):
        return self._description or None

    @description.setter
    def description(self, value):
        self._description = value

    def get_data_format(self):
        raise NotImplementedError(f'{self.__class__.__name__} does not implement get_data_format')
