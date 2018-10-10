import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dotenv import load_dotenv

from .base import Segmentation2DModelBase
from utils import MetricClass
from .utils import (
    weighted_binary_cross_entropy,
    soft_dice_score,
    normalize_image,
    ImageAugmentor,
    get_2d_from_3d,
    co_shuffle,
    get_tensor_from_array,
)

load_dotenv('./.env')
RESULT_DIR_BASE = os.environ.get('RESULT_DIR')


class ToyModel(Segmentation2DModelBase):
    def __init__(
            self,
            channels: int = 1,
            depth: int = 200,
            height: int = 200,
            width: int = 200,
            metadata_dim: int = 0,
            num_units: [int] = (32, 32, 64, 64, 128),
            pooling_layer_num: [int] = (1, 3),
            kernel_size: int = 3,
            lr: float = 1e-4,
        ):
        self.num_units = num_units
        self.data_channels = channels
        self.data_depth = depth
        self.data_height = height
        self.data_width = width
        self.metadata_dim = metadata_dim
        self.comet_experiment = None

        EXP_ID = os.environ.get('EXP_ID')
        self.result_path = os.path.join(RESULT_DIR_BASE, EXP_ID)

        self.image_augmentor = ImageAugmentor(
            channels,
            height,
            width,
            mode='albumentations',
        )

        self.model = ToyModelNet(
            image_chns=self.data_channels,
            image_height=self.data_height,
            image_width=self.data_width,
            num_units=num_units,
            pooling_layer_num=pooling_layer_num,
            kernel_size=kernel_size,
        )
        self.opt = optim.Adam(params=self.model.parameters(), lr=lr)
        if torch.cuda.is_available():
            self.model.cuda()

    def fit_generator(self, training_data_generator, validation_data_generator, **kwargs):
        print(kwargs)
        batch_size = kwargs['batch_size']
        epoch_num = kwargs['epoch_num']

        verbose_epoch_num = kwargs['verbose_epoch_num']
        if 'experiment' in kwargs:
            self.comet_experiment = kwargs['experiment']

        for i_epoch in range(epoch_num):
            losses, dice_scores = self.train_on_batch(training_data_generator, batch_size)
            if i_epoch % verbose_epoch_num == 0:
                print(
                    f'epoch: {i_epoch}',
                    f', bce_loss: {np.mean(losses)}',
                    f', dice_score: {np.mean(dice_scores)}',
                )

                metrics = self._validate(
                    validation_data_generator, batch_size, verbose_epoch_num // 10
                )
                if self.comet_experiment is not None:
                    self.comet_experiment.log_multiple_metrics({
                        'bce_loss': np.mean(losses),
                        'dice_score': np.mean(dice_scores),
                    },
                        prefix='training')
                    self.comet_experiment.log_multiple_metrics(
                        metrics, prefix='validation', step=i_epoch
                    )

    def train_on_batch(self, training_data_generator, batch_size):
        image, label = self._get_data_with_generator(
            training_data_generator,
            1,
        )
        losses = []
        dice_scores = []

        for batch_idx in range(self.data_depth // batch_size):
            self.model.zero_grad()
            batch_image = image[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_label = label[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            batch_image = get_tensor_from_array(batch_image)
            batch_label = get_tensor_from_array(batch_label)

            pred = self.model(batch_image)
            bce_loss = weighted_binary_cross_entropy(pred, batch_label, weights=(1, 1. / 1e-3))
            dice_score = soft_dice_score(pred, batch_label)

            # total_loss = bce_loss
            # total_loss = bce_loss - dice_score
            # total_loss = 1 - dice_score
            total_loss = bce_loss - torch.log(dice_score)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.opt.step()

            losses.append(bce_loss.item())
            dice_scores.append(dice_score.item())

        return losses, dice_scores

    def _validate(self, validation_data_generator, batch_size, verbose_epoch_num):
        label_buff = []
        pred_buff = []

        for batch_num in range(verbose_epoch_num):
            image, label = self._get_data_with_generator(
                validation_data_generator,
                1,
            )
            pred = self._predict_on_2d_images(image, batch_size)
            label_buff.extend(label)
            pred_buff.extend(pred)

        label = np.asarray(label_buff)
        pred = np.asarray(pred_buff)
        return MetricClass(pred, label).all_metrics()

    def _get_data_with_generator(self, generator, batch_size):
        batch_data = generator(batch_size=batch_size)
        batch_volume, batch_label = batch_data['img'], batch_data['label']

        batch_image = get_2d_from_3d(batch_volume)
        batch_label = get_2d_from_3d(batch_label)
        batch_image = normalize_image(batch_image)

        batch_image, batch_label = co_shuffle(batch_image, batch_label)
        batch_image, batch_label = self.image_augmentor.co_transform(batch_image, batch_label)
        return batch_image, batch_label

    def _predict_on_2d_images(self, image, batch_size):
        pred_buff = []
        self.model.eval()

        batch_num = math.ceil(image.shape[0] // batch_size)
        for batch_idx in range(batch_num):
            end_index = min(image.shape[0], (batch_idx + 1) * batch_size)
            batch_image = image[batch_idx * batch_size: end_index]
            batch_image = get_tensor_from_array(batch_image)

            batch_pred = self.model(batch_image)
            batch_pred = batch_pred.cpu().data.numpy()
            pred_buff.extend(batch_pred)

        self.model.train()
        pred_buff = np.asarray(pred_buff)
        return pred_buff

    def predict(self, test_volumes, **kwargs):
        batch_size = kwargs['batch_size']
        test_images = get_2d_from_3d(test_volumes)
        test_images = normalize_image(test_images)
        pred = self._predict_on_2d_images(test_images, batch_size)
        return pred

    def save(self):
        torch.save(self.model, os.path.join(self.result_path, 'model'))
        print(f'model save to {self.result_path}')

    def load(self, file_path):
        self.model = torch.load(os.path.join(file_path, 'model'))
        print(f'model loaded from {file_path}')


class ToyModelNet(nn.Module):
    def __init__(
        self,
        image_chns,
        image_height,
        image_width,
        num_units,
        kernel_size,
        pooling_layer_num
    ):
        super(ToyModelNet, self).__init__()
        self.image_chns = image_chns
        self.image_height = image_height
        self.image_width = image_width

        encoder_num_units = (image_chns,) + num_units
        self.encoder_convs = nn.ModuleList()
        self.encoder_batchnorms = nn.ModuleList()
        for idx in range(len(encoder_num_units) - 1):
            if idx in pooling_layer_num:
                stride = 2
            else:
                stride = 1
            in_chns = encoder_num_units[idx]
            out_chns = encoder_num_units[idx + 1]
            conv = nn.Conv2d(
                in_chns,
                out_chns,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
            batchnorm = nn.BatchNorm2d(in_chns)
            self.encoder_convs.append(conv)
            self.encoder_batchnorms.append(batchnorm)

        decoder_num_units = num_units[::-1] + (image_chns,)
        self.decoder_deconvs = nn.ModuleList()
        self.decoder_batchnorms = nn.ModuleList()
        img_size = image_height // (2 ** len(pooling_layer_num))
        for idx in range((len(decoder_num_units)) - 1):
            if idx in pooling_layer_num:
                stride = 2
                img_size = img_size * 2
                output_padding = 1
            else:
                stride = 1
                output_padding = 0
            deconv = nn.ConvTranspose2d(
                decoder_num_units[idx],
                decoder_num_units[idx + 1],
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                output_padding=output_padding,
            )
            batchnorm = nn.BatchNorm2d(decoder_num_units[idx])
            self.decoder_deconvs.append(deconv)
            self.decoder_batchnorms.append(batchnorm)

    def forward(self, inp):
        x = inp
        for conv, batchnorm in zip(self.encoder_convs, self.encoder_batchnorms):
            x = batchnorm(x)
            x = F.relu(x)
            x = conv(x)

        for deconv, batchnorm in zip(self.decoder_deconvs, self.decoder_batchnorms):
            x = batchnorm(x)
            x = F.relu(x)
            x = deconv(x)

        x = torch.sigmoid(x)
        return x
