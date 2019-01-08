from .base import BatchSamplerBase


class ThreeDimBatchSampler(BatchSamplerBase):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def convert_to_feedable(self, batch_data, batch_size, training=False, **kwargs):

        volume = batch_data['volume']
        label = batch_data['label']

        feedable_data_list = [volume]
        feedable_label_list = [label]

        return feedable_data_list, feedable_label_list

    def reassemble(self, preds: list, test_data: dict, **kwargs):
        volume_segmentations = preds[0]
        return volume_segmentations
