import numpy as np
from albumentations import (
    Compose,
    ShiftScaleRotate,
    Flip,
    # ElasticTransform,
    # RandomBrightnessContrast,
)

from .base import AugmentorBase


class ImageAugmentor(AugmentorBase):

    def __init__(
        self,
        data_format: dict,
        mode='albumentations',
    ):
        self.data_channels = data_format['channels']
        self.data_height = data_format['height']
        self.data_width = data_format['width']
        self.mode = mode

        self.transform_fns = {
            'albumentations': self._albumentations_transform,
            'identity': self._identity_transform,
        }

    def co_transform(self, batch_image, batch_label, **kwargs):
        assert(batch_image.shape[-3:] == (self.data_channels, self.data_height, self.data_width))
        assert(len(batch_label) == len(batch_image))

        transform_fn = self.transform_fns[self.mode]
        return transform_fn(batch_image, batch_label)

    def _albumentations_transform(self, batch_image: np.array, batch_label: np.array):
        batch_image = np.transpose(batch_image, (0, 2, 3, 1))

        data_dict = {'image': batch_image[0], 'mask': batch_label[0]}
        header_dict = {}
        for idx, (image, label) in enumerate(zip(batch_image, batch_label)):
            data_dict[f'image_{idx}'] = image
            data_dict[f'mask_{idx}'] = label
            header_dict[f'image_{idx}'] = 'image'
            header_dict[f'mask_{idx}'] = 'mask'

        augmentation = self._albumentations_strong_aug()
        augmentation.add_targets(header_dict)
        augmented_data = augmentation(**data_dict)

        transformed_image = [
            augmented_data[f'image_{idx}'] for idx in range(len(batch_image))
        ]
        transformed_label = [
            augmented_data[f'mask_{idx}'] for idx in range(len(batch_label))
        ]

        transformed_image = np.asarray(transformed_image)
        transformed_label = np.asarray(transformed_label)
        transformed_image = np.transpose(transformed_image, (0, 3, 1, 2))
        return transformed_image, transformed_label

    @staticmethod
    def _albumentations_strong_aug() -> Compose:
        return Compose([
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20),
            Flip(),
            # ElasticTransform(alpha=720, sigma=24, approximate=False),
            # RandomBrightnessContrast(),
        ])

    @staticmethod
    def _identity_transform(batch_image, batch_label):
        return batch_image, batch_label
