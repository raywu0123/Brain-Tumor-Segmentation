from .base import AugmentorBase
from .image_augmentation import ImageAugmentor
from utils import get_3d_from_2d, get_2d_from_3d


class DepthwiseVolumeAugmentor(AugmentorBase):

    def __init__(self, data_format: dict):
        self.data_format = data_format
        self.image_augmentor = ImageAugmentor(data_format)

    def co_transform(self, volume, label, **kwargs):
        if volume.ndim != 5:
            raise ValueError(f'batch_data is not a volume.')

        if label.ndim != 5:
            raise ValueError(f'batch_label is not a volume.')

        volume_2d = get_2d_from_3d(volume)
        label_2d = get_2d_from_3d(label)

        augmented_volume_2d, augmented_label_2d = \
            self.image_augmentor.co_transform(volume_2d, label_2d)
        augmented_volume = get_3d_from_2d(augmented_volume_2d, data_depth=self.data_format['depth'])
        augmented_label = get_3d_from_2d(augmented_label_2d, data_depth=self.data_format['depth'])
        return augmented_volume, augmented_label
