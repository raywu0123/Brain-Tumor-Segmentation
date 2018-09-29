class Segmentation2DModelBase:
    def fit(self, training_data, validation_data, **kwargs):
        raise NotImplementedError('fit not implemented')

    def fit_generator(self, training_data_generator, validation_data_generator, **kwargs):
        raise NotImplementedError('fit_generator not implemented')

    def predict(self, x_test, **kwargs):
        raise NotImplementedError('predict not implemented')
