import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import Augmentor
from albumentations import (
    Compose,
    ShiftScaleRotate,
    Flip,
    ElasticTransform,
)
import imgaug as ia
from imgaug import augmenters as iaa

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
        self.keras_data_generator = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            data_format='channels_first',
            # brightness_range=(0.8, 1.2),
        )
        self.imgaug_seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.PiecewiseAffine(scale=0.01),
            iaa.Affine(
                scale=(0.9, 1.1),
                translate_percent=0.1,
                rotate=(-20, 20),
                shear=(-5, 5),
                mode=ia.ALL,
            ),
        ])
        self.transform_fns = {
            'keras': self._keras_transform,
            'imgaug': self._imgaug_transform,
            'Augmentor': self._Augmentor_transform,
            'albumentations': self._albumentations_transform,
        }

    def co_transform(self, batch_image, batch_label, **kwargs):
        assert(batch_image.shape[-3:] == (self.data_channels, self.data_height, self.data_width))
        assert(len(batch_label) == len(batch_image))

        transform_fn = self.transform_fns[self.mode]
        return transform_fn(batch_image, batch_label)

    def _keras_transform(self, batch_image, batch_label):
        transformed_image = []
        transformed_label = []
        for image, label in zip(batch_image, batch_label):
            rdm_transform = self.keras_data_generator.get_random_transform(
                (self.data_channels, self.data_height, self.data_width),
            )
            image = self.keras_data_generator.apply_transform(image, rdm_transform)
            label = self.keras_data_generator.apply_transform(label, rdm_transform)
            transformed_image.append(image)
            transformed_label.append(label)

        transformed_image = np.asarray(transformed_image)
        transformed_label = np.asarray(transformed_label)
        return transformed_image, transformed_label

    def _imgaug_transform(self, batch_image, batch_label):
        batch_image = np.transpose(batch_image, (0, 2, 3, 1))
        batch_label = np.transpose(batch_label, (0, 2, 3, 1))

        seq_det = self.imgaug_seq.to_deterministic()
        batch_image = seq_det.augment_images(batch_image)
        batch_label = seq_det.augment_images(batch_label)

        batch_image = np.transpose(batch_image, (0, 3, 1, 2))
        batch_label = np.transpose(batch_label, (0, 3, 1, 2))
        return batch_image, batch_label

    def _Augmentor_transform(self, batch_image, batch_label):
        # currently not available
        batch_image = np.transpose(batch_image, (0, 2, 3, 1))
        batch_label = np.transpose(batch_label, (0, 2, 3, 1))

        images = list(zip(batch_image, batch_label))
        p = Augmentor.DataPipeline(images)
        p.rotate(1, max_left_rotation=20, max_right_rotation=20)
        p.flip_top_bottom(0.5)

        augmented_images = p.sample(len(batch_image))
        unzipped_images_list = list(zip(*augmented_images))
        batch_image = np.asarray(unzipped_images_list[0])
        batch_label = np.asarray(unzipped_images_list[1])

        batch_image = np.transpose(batch_image, (0, 3, 1, 2))
        batch_label = np.transpose(batch_label, (0, 3, 1, 2))
        return batch_image, batch_label

    def _albumentations_transform(self, batch_image, batch_label):
        batch_image = np.transpose(batch_image, (0, 2, 3, 1))
        batch_label = np.transpose(batch_label, (0, 2, 3, 1))
        transformed_image = []
        transformed_label = []
        for image, label in zip(batch_image, batch_label):
            augmentation = self._albumentations_strong_aug()
            augmented_data = augmentation(image=image, mask=label)
            augmented_image, augmented_label = augmented_data['image'], augmented_data['mask']
            augmented_image = np.reshape(augmented_image, image.shape)
            augmented_label = np.reshape(augmented_label, label.shape)
            transformed_image.append(augmented_image)
            transformed_label.append(augmented_label)

        transformed_image = np.asarray(transformed_image)
        transformed_label = np.asarray(transformed_label)
        transformed_image = np.transpose(transformed_image, (0, 3, 1, 2))
        transformed_label = np.transpose(transformed_label, (0, 3, 1, 2))
        return transformed_image, transformed_label

    @staticmethod
    def _albumentations_strong_aug():
        return Compose([
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20),
            Flip(),
            ElasticTransform(alpha=720, sigma=24),
        ])
