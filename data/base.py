from utils import MetricClass


class DataInterface:

    _description = None
    _metric = MetricClass

    def get_training_dataloader(self, batch_size, shuffle, num_workers):
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement training_datagenerator'
        )

    def get_testing_dataloader(self, batch_size, shuffle, num_workers):
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement testing_datagenerator'
        )

    @property
    def training_datagenerator(self):
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement training_datagenerator'
        )

    @property
    def testing_datagenerator(self):
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement testing_datagenerator'
        )

    def get_training_data(self):
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement get_training_data'
        )

    def get_testing_data(self):
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement get_testing_data'
        )

    def get_all_data(self):
        raise NotImplementedError(
            f'{self.__class__.__name__} does not implement get_all_data'
        )

    @property
    def metric(self):
        return self._metric or None

    @property
    def description(self):
        return self._description or None

    @description.setter
    def description(self, value):
        self._description = value

    def get_data_format(self):
        raise NotImplementedError(f'{self.__class__.__name__} does not implement get_data_format')
