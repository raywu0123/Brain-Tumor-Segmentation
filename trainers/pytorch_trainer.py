from abc import ABC
import os

import torch
import comet_ml
from dotenv import load_dotenv

from .base import TrainerBase

load_dotenv('./.env')
RESULT_DIR_BASE = os.environ.get('RESULT_DIR')


class PytorchTrainer(TrainerBase, ABC):

    def __init__(
            self,
            model: torch.nn.Module,
            comet_experiment: comet_ml.Experiment = None,
            lr: float = 1e-4,
    ):
        self.comet_experiment = comet_experiment
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()

        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.i_epoch = 0
        EXP_ID = os.environ.get('EXP_ID')
        self.result_path = os.path.join(RESULT_DIR_BASE, EXP_ID)

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
                    self.comet_experiment.log_multiple_metrics(
                        log_dict, prefix='training', step=self.i_epoch
                    )
                    self.comet_experiment.log_multiple_metrics(
                        metrics, prefix='validation', step=self.i_epoch
                    )
