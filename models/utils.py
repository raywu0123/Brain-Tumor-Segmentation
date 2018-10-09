import torch
# from torch import nn
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


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
    assert(pred == tar.shape)

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


def get_image_from_volume(batch_volume):
    assert(batch_volume.ndim == 5)
    batch_volume = np.transpose(batch_volume, (4, 0, 1, 2, 3))
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
    ):
        self.data_channels = data_channels
        self.data_height = data_height
        self.data_width = data_width
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

    def co_transform(self, batch_image, batch_label):
        transformed_image = []
        transformed_label = []
        for image, label in zip(batch_image, batch_label):
            rdm_transform = self.keras_data_generator.get_random_transform(
                (self.data_channels, self.data_height, self.data_width),
            )
            image = self.image_data_generator.apply_transform(image, rdm_transform)
            label = self.image_data_generator.apply_transform(label, rdm_transform)
            transformed_image.append(image)
            transformed_label.append(label)

        transformed_image = np.asarray(transformed_image)
        transformed_label = np.asarray(transformed_label)
        return transformed_image, transformed_label
