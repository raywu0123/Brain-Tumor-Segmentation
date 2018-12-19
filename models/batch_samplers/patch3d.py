from .base import BatchSamplerBase


class Patch3DBatchSampler(BatchSamplerBase):

    def convert_to_feedable(self, batch_data, batch_size, training, **kwargs):
        pass

    def reassemble(self, prediction: list, test_data: list, **kwargs):
        pass
