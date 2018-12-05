import math
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from .utils import (
    weighted_cross_entropy,
    soft_dice_score,
    normalize_image,
    get_2d_from_3d,
    get_3d_from_2d,
    co_shuffle,
    get_tensor_from_array,
)


class ModelBase(ABC):

    @abstractmethod
    def fit_generator(self, training_data_generator, validation_data_generator, **kwargs):
        pass

    @abstractmethod
    def predict(self, test_data, **kwargs):
        pass


class PytorchModelBase(ModelBase, nn.Module):

    pass


class Model2DBase(PytorchModelBase):

    def __init__(self):
        super(Model2DBase, self).__init__()

    def fit_generator(self, training_data_generator, batch_size, **kwargs):
        image, label = self._get_data_with_generator(training_data_generator)
        crossentropy_losses = []
        dice_scores = []

        for batch_idx in range(self.data_format['depth'] // batch_size):
            self.model.zero_grad()
            batch_image = image[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_label = label[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            class_weights = np.divide(
                1., np.mean(batch_label, axis=(0, 2, 3)),
                out=np.ones(batch_label.shape[1]),
                where=np.mean(batch_label, axis=(0, 2, 3)) != 0,
            )
            batch_image = get_tensor_from_array(batch_image)
            batch_label = get_tensor_from_array(batch_label)

            pred = self.model(batch_image)
            crossentropy_loss = weighted_cross_entropy(pred, batch_label, weights=class_weights)
            dice_score = soft_dice_score(pred, batch_label)
            total_loss = crossentropy_loss - torch.log(dice_score)

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.opt.step()

            crossentropy_losses.append(crossentropy_loss.item())
            dice_scores.append(dice_score.item())

        crossentropy_losses = np.mean(crossentropy_losses)
        dice_scores = np.mean(dice_scores)
        return {'crossentropy_loss': crossentropy_losses, 'soft_dice': dice_scores}

    @staticmethod
    def _get_data_with_generator(generator):
        batch_data = generator(batch_size=1)
        batch_volume, batch_label = batch_data['volume'], batch_data['label']

        batch_image = get_2d_from_3d(batch_volume)
        batch_label = get_2d_from_3d(batch_label)

        batch_image, batch_label = co_shuffle(batch_image, batch_label)
        return batch_image, batch_label

    def _predict_on_2d_images(self, image, batch_size, verbose=False):
        pred_buff = []
        self.model.eval()
        batch_num = math.ceil(image.shape[0] / batch_size)
        iterator = list(range(batch_num))
        if verbose:
            iterator = tqdm(iterator)

        for batch_idx in iterator:
            end_index = min(image.shape[0], (batch_idx + 1) * batch_size)
            batch_image = image[batch_idx * batch_size: end_index]
            batch_image = get_tensor_from_array(batch_image)

            batch_pred = self.model(batch_image)
            batch_pred = batch_pred.cpu().data.numpy()
            pred_buff.extend(batch_pred)

        self.model.train()
        pred_buff = np.asarray(pred_buff)
        return pred_buff

    def predict(self, test_data, **kwargs):
        print(kwargs)
        test_volumes = test_data['volume']
        test_images = get_2d_from_3d(test_volumes)
        test_images = normalize_image(test_images)
        pred_images = self._predict_on_2d_images(test_images, **kwargs)
        pred_volumes = get_3d_from_2d(pred_images, self.data_format['depth'])
        return pred_volumes
