from .base import AugmentorBase
from .image_augmentation import ImageAugmentor
from utils import get_3d_from_2d, get_2d_from_3d


class VolumeAugmentor(AugmentorBase):

    def __init__(self, data_format: dict, **kwargs):
        self.data_format = data_format
        self.image_augmentor_1 = ImageAugmentor(data_format, **kwargs)
        self.image_augmentor_2 = ImageAugmentor({
            **data_format,
            'height': data_format['depth'],
            'depth': data_format['height'],
        }, **kwargs)

    @staticmethod
    def _single_transform(volume, label, augmentor, data_depth):
        volume_2d = get_2d_from_3d(volume)
        label_2d = get_2d_from_3d(label)
        augmented_volume_2d, augmented_label_2d = \
            augmentor.co_transform(volume_2d, label_2d)

        augmented_volume = get_3d_from_2d(augmented_volume_2d, data_depth=data_depth)
        augmented_label = get_3d_from_2d(augmented_label_2d, data_depth=data_depth)
        return augmented_volume, augmented_label

    def co_transform(self, volume, label, **kwargs):
        if volume.ndim != 5:
            raise ValueError(f'batch_data is not a valid volume. shape: {volume.shape}')

        if label.ndim != 4:
            raise ValueError(f'batch_label is not a valid label. shape: {label.shape}')

        augmented_volume, augmented_label = self._single_transform(
            volume,
            label,
            self.image_augmentor_1,
            self.data_format['depth'],
        )

        augmented_volume = augmented_volume.transpose([0, 1, 3, 2, 4])
        augmented_label = augmented_label.transpose([0, 2, 1, 3])

        augmented_volume, augmented_label = self._single_transform(
            augmented_volume,
            augmented_label,
            self.image_augmentor_2,
            self.data_format['height'],
        )

        augmented_volume = augmented_volume.transpose([0, 1, 3, 2, 4])
        augmented_label = augmented_label.transpose([0, 2, 1, 3])
        return augmented_volume, augmented_label
