import os
import math
import multiprocessing as mp
import time

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

from .utils import (
    weighted_cross_entropy,
    soft_dice_score,
    normalize_image,
    ImageAugmentor,
    get_2d_from_3d,
    get_3d_from_2d,
    co_shuffle,
    get_tensor_from_array,
)


load_dotenv('./.env')
RESULT_DIR_BASE = os.environ.get('RESULT_DIR')


class ModelBase:
    def __init__(
            self,
            channels: int = 1,
            depth: int = 200,
            height: int = 200,
            width: int = 200,
            metadata_dim: int = 0,
            class_num: int = 2,
        ):
        self.data_channels = channels
        self.data_depth = depth
        self.data_height = height
        self.data_width = width
        self.metadata_dim = metadata_dim
        self.class_num = class_num
        self.comet_experiment = None
        self.model = None
        self.opt = None
        EXP_ID = os.environ.get('EXP_ID')
        self.result_path = os.path.join(RESULT_DIR_BASE, EXP_ID)

    def fit(self, training_data, validation_data, **kwargs):
        raise NotImplementedError('fit not implemented')

    def fit_dataloader(self, get_training_dataloader, get_validation_dataloader, **kwargs):
        raise NotImplementedError('fit_dataloader not implemented')

    def train_on_batch(self, training_datagenerator, batch_size):
        raise NotImplementedError('fit_dataloader not implemented')

    def predict(self, test_data, batch_size, **kwargs):
        raise NotImplementedError('predict not implemented')

    def save(self):
        torch.save(self.model, os.path.join(self.result_path, 'model'))
        print(f'model save to {self.result_path}')

    def load(self, file_path):
        self.model = torch.load(os.path.join(file_path, 'model'))
        print(f'model loaded from {file_path}')

    def _validate(self, validation_datagenerator, batch_size, metric):
        batch_data = validation_datagenerator(batch_size=1)
        label = batch_data['label']
        pred = self.predict(batch_data, batch_size)
        return metric(pred, label).all_metrics()

    def fit_generator(self, training_datagenerator, validation_datagenerator, metric, **kwargs):
        print(kwargs)
        batch_size = kwargs['batch_size']
        epoch_num = kwargs['epoch_num']
        verbose_epoch_num = kwargs['verbose_epoch_num']
        if 'experiment' in kwargs:
            self.comet_experiment = kwargs['experiment']
        for i_epoch in range(epoch_num):
            loss, dice_score = self.train_on_batch(training_datagenerator, batch_size)
            if i_epoch % verbose_epoch_num == 0:
                print(
                    f'epoch: {i_epoch}',
                    f', crossentropy_loss: {loss}',
                    f', dice_score: {dice_score}',
                )

                self.save()
                metrics = self._validate(
                    validation_datagenerator, batch_size, metric,
                )
                if self.comet_experiment is not None:
                    self.comet_experiment.log_multiple_metrics({
                        'crossentropy_loss': loss,
                        'dice_score': dice_score,
                    }, prefix='training', step=i_epoch
                    )
                    self.comet_experiment.log_multiple_metrics(
                        metrics, prefix='validation', step=i_epoch
                    )


class Model2DBase(ModelBase):
    def __init__(
            self,
            channels: int = 1,
            depth: int = 200,
            height: int = 200,
            width: int = 200,
            metadata_dim: int = 0,
            class_num: int = 2,
        ):
        super(Model2DBase, self).__init__(
            channels=channels,
            depth=depth,
            height=height,
            width=width,
            metadata_dim=metadata_dim,
            class_num=class_num,
        )

        self.image_augmentor = ImageAugmentor(
            channels,
            height,
            width,
            mode='albumentations',
        )

    def train_on_batch(self, training_datagenerator, batch_size):
        image, label = self._get_augmented_image_and_label(datagenerator=training_datagenerator)
        losses = []
        dice_scores = []

        for batch_idx in range(self.data_depth // batch_size):
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

            losses.append(crossentropy_loss.item())
            dice_scores.append(dice_score.item())

        losses = np.mean(losses)
        dice_scores = np.mean(dice_scores)

        return losses, dice_scores

    def _get_augmented_image_and_label(self, **kwargs):
        datagenerator = kwargs['datagenerator']
        image, label = self._get_data_with_generator(
            datagenerator,
            1,
        )
        image, label = self.image_augmentor.co_transform(image, label)
        return image, label

    @staticmethod
    def _get_data_with_generator(generator, batch_size):
        batch_data = generator(batch_size=batch_size)
        batch_volume, batch_label = batch_data['volume'], batch_data['label']

        batch_image = get_2d_from_3d(batch_volume)
        batch_label = get_2d_from_3d(batch_label)

        batch_image = normalize_image(batch_image)

        batch_image, batch_label = co_shuffle(batch_image, batch_label)
        return batch_image, batch_label

    def _predict_on_2d_images(self, image, batch_size, verbose=False, **kwargs):
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

    def predict(self, test_data, batch_size, **kwargs):
        print(kwargs)
        test_volumes = test_data['volume']
        test_images = get_2d_from_3d(test_volumes)
        test_images = normalize_image(test_images)
        pred_images = self._predict_on_2d_images(test_images, batch_size, **kwargs)
        pred_volumes = get_3d_from_2d(pred_images, self.data_depth)
        return pred_volumes


class AsyncModel2DBase(Model2DBase):
    def __init__(
        self,
        channels: int = 1,
        depth: int = 200,
        height: int = 200,
        width: int = 200,
        metadata_dim: int = 0,
        class_num: int = 2,
    ):
        super().__init__(
            channels,
            depth,
            height,
            width,
            metadata_dim,
            class_num,
        )
        self.data_queue = mp.Queue()
        self.datagenerator = None

    def fit_generator(self, training_datagenerator, validation_datagenerator, metric, **kwargs):
        print(kwargs)
        batch_size = kwargs['batch_size']
        epoch_num = kwargs['epoch_num']

        verbose_epoch_num = kwargs['verbose_epoch_num']
        if 'experiment' in kwargs:
            self.comet_experiment = kwargs['experiment']

        self.datagenerator = training_datagenerator
        process = mp.Process(target=self._put_data_into_queue)
        process.start()

        for i_epoch in range(epoch_num):
            losses, dice_scores = self.train_on_batch(training_datagenerator, batch_size)
            if i_epoch % verbose_epoch_num == 0:
                print(
                    f'epoch: {i_epoch}',
                    f', bce_loss: {np.mean(losses)}',
                    f', dice_score: {np.mean(dice_scores)}',
                )

                self.save()
                metrics = self._validate(
                    validation_datagenerator, batch_size, verbose_epoch_num // 10, metric,
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

    def _get_augmented_image_and_label(self, **kwargs):
        while self.data_queue.empty():
            time.sleep(0.1)
        image, label = self.data_queue.get()
        return image, label

    def _put_data_into_queue(self):
        while True:
            if self.data_queue.qsize() < 10:
                image, label = self._get_data_with_generator(
                    self.datagenerator,
                    1,
                )
                image, label = self.image_augmentor.co_transform(image, label)
                self.data_queue.put((image, label))
