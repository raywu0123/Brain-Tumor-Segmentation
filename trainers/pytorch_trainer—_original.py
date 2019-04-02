from abc import ABC
import os

import comet_ml
import torch
from dotenv import load_dotenv
import numpy as np
from tqdm import tqdm


from .base import TrainerBase
from preprocess_tools.image_utils import save_array_to_nii

load_dotenv('./.env')
RESULT_DIR_BASE = os.environ.get('RESULT_DIR')


class PytorchTrainer(TrainerBase, ABC):

    def __init__(
            self,
            model: torch.nn.Module,
            comet_experiment: comet_ml.Experiment = None,
            checkpoint_dir=None,
            lr: float = 1e-4,
    ):
        self.comet_experiment = comet_experiment

        self.model = model
        print(f'Total parameters: {self.count_parameters()}')
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        if checkpoint_dir is not None:
            self.load(checkpoint_dir)

        if torch.cuda.is_available():
            self.model.cuda()

        self.i_epoch = 0
        EXP_ID = os.environ.get('EXP_ID')
        self.result_path = os.path.join(RESULT_DIR_BASE, EXP_ID)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def save(self):
        torch.save(
            {
                'epoch': self.i_epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.opt.state_dict(),
            },
            os.path.join(self.result_path, 'checkpoint.pth.tar')
        )
        print(f'model saved to {self.result_path}')

    def load(self, file_path):
        checkpoint = torch.load(os.path.join(file_path, 'checkpoint.pth.tar'))
        self.model.load_state_dict(checkpoint['state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer'])
        self.i_epoch = checkpoint['epoch'] + 1
        print(f'model loaded from {file_path}')

    def _validate(self, validation_data_generator, metric, **kwargs):
        batch_data = validation_data_generator(batch_size=1)
        label = batch_data['label']
        pred = self.model.predict(batch_data, **kwargs)
        return metric(pred, label).all_metrics()

    def fit_generator(self, training_data_generator, validation_data_generator, metric, **kwargs):
        print(kwargs)
        batch_size = kwargs['batch_size']
        epoch_num = kwargs['epoch_num']
        verbose_epoch_num = kwargs['verbose_epoch_num']

        for self.i_epoch in range(self.i_epoch, self.i_epoch + epoch_num):
            log_dict = self.model.fit_generator(
                training_data_generator, self.opt, batch_size=batch_size
            )

            if self.i_epoch % verbose_epoch_num == 0:
                print(f'epoch: {self.i_epoch}', log_dict)
                self.save()
                metrics = self._validate(
                    validation_data_generator, metric, batch_size=batch_size
                )
                if self.comet_experiment is not None:
                    self.comet_experiment.log_metrics(
                        log_dict, prefix='training', step=self.i_epoch
                    )
                    self.comet_experiment.log_metrics(
                        metrics, prefix='validation', step=self.i_epoch
                    )

    def predict_on_generator(self, data_generator, save_base_dir, metric, **kwargs):
        prob_prediction_path = os.path.join(save_base_dir, f'prob_predict')
        hard_prediction_path = os.path.join(save_base_dir, f'hard_predict')
        if not os.path.exists(save_base_dir):
            os.mkdir(save_base_dir)
        if not os.path.exists(prob_prediction_path):
            os.mkdir(prob_prediction_path)
        if not os.path.exists(hard_prediction_path):
            os.mkdir(hard_prediction_path)

        metrics_dict = {}

        print(f'predicting on {len(data_generator)} volumes...')
        for _ in tqdm(range(len(data_generator))):
            batch_data = data_generator(batch_size=1)
            label, data_id = batch_data['label'], batch_data['data_ids'][0]
            pred = self.model.predict(batch_data, **kwargs)

            metrics = metric(pred, label).all_metrics(verbose=False)
            metrics_dict[data_id] = metrics

            # to [D, H, W, C] format
            pred = pred[0].transpose([2, 3, 1, 0])
            hard_pred = np.argmax(pred, axis=-1)

            data_id = batch_data['data_ids'][0]
            affine = batch_data['affines'][0]

            save_array_to_nii(pred, os.path.join(prob_prediction_path, data_id), affine)
            save_array_to_nii(hard_pred, os.path.join(hard_prediction_path, data_id), affine)

        return metrics_dict
