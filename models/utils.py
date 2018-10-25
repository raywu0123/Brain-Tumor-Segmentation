import torch
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import Augmentor
from albumentations import (
    Compose,
    OneOf,
    NoOp,
    ShiftScaleRotate,
    Flip,
    ElasticTransform,
)


def weighted_binary_cross_entropy(output, target, weights=None):
    epsilon = 1e-8
    if weights is not None:
        assert len(weights) == 2
        weights = torch.tensor(weights, requires_grad=False)
        if torch.cuda.is_available:
            weights = weights.cuda()
        loss = weights[1] * (target * torch.log(output + epsilon)) + \
            weights[0] * ((1 - target) * torch.log(1 - output + epsilon))
    else:
        loss = target * torch.log(output + epsilon) + (1 - target) * torch.log(1 - output + epsilon)

    return torch.neg(torch.mean(loss))


def soft_dice_score(pred, tar):
    # Calculated for whole batch
    assert(pred.shape == tar.shape)

    batch_size = pred.shape[0]
    m1 = pred.view(batch_size, -1)
    m2 = tar.view(batch_size, -1)
    intersection = (m1 * m2)

    m1 = torch.sum(m1)
    m2 = torch.sum(m2)
    intersection = torch.sum(intersection)

    dice_score = (2. * intersection + 1) / (m1 + m2 + 1)
    return dice_score


def normalize_image(batch_image):
    assert(batch_image.ndim == 4)
    std = np.std(batch_image, axis=(1, 2, 3), keepdims=True)
    std_is_zero = std == 0
    batch_image = (batch_image - np.mean(batch_image, axis=(1, 2, 3), keepdims=True)) \
        / (std + std_is_zero.astype(float))
    return batch_image


def get_2d_from_3d(batch_volume):
    assert(batch_volume.ndim == 5)
    batch_volume = np.transpose(batch_volume, (0, 4, 1, 2, 3))
    batch_image = batch_volume.reshape(-1, *batch_volume.shape[-3:])
    return batch_image


def co_shuffle(batch_data, batch_label):
    assert(batch_data.shape == batch_label.shape)
    p = np.random.permutation(len(batch_data))
    batch_data = batch_data[p]
    batch_label = batch_label[p]
    return batch_data, batch_label


def get_tensor_from_array(array):
    tensor = torch.Tensor(array)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


class ImageAugmentor:
    def __init__(
        self,
        data_channels: int,
        data_height: int,
        data_width: int,
        mode='keras'
    ):
        self.data_channels = data_channels
        self.data_height = data_height
        self.data_width = data_width
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

    def co_transform(self, batch_image, batch_label):
        assert(batch_image.shape[-3:] == (self.data_channels, self.data_height, self.data_width))
        assert(batch_label.shape == batch_image.shape)

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
            OneOf(
                [ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=20), NoOp()]
            ),
            OneOf(
                [Flip(), NoOp()]
            ),
            OneOf(
                [ElasticTransform(alpha=720, sigma=24), NoOp()]
            ),
        ])
