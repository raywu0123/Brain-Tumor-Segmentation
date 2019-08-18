from abc import ABC, abstractmethod

import torch
from torch import nn

from .batch_samplers import BatchSamplerHub
from .utils import summarize_logs


class ModelBase(ABC):

    @abstractmethod
    def fit_generator(self, training_data_generator, optimizer, **kwargs):
        pass

    @abstractmethod
    def predict(self, test_data, **kwargs):
        pass


class PytorchModelBase(ModelBase, nn.Module):

    def __init__(self, batch_sampler_id: str, loss_fn, data_format: dict):
        nn.Module.__init__(self)
        self.loss_fn = loss_fn
        self.batch_sampler_constructor = BatchSamplerHub[batch_sampler_id]
        self.batch_sampler = self.batch_sampler_constructor(
            data_format=data_format
        )

    def fit_generator(self, training_data_generator, optimizer, batch_size, **kwargs):
        """
        fit on generator for one single volume
        """
        self.train()
        data = training_data_generator(batch_size=1)
        # batch_size here stands for number of volumes
        # most devices can only hold one singe volume

        batch_data_list, batch_label_list = self.batch_sampler.convert_to_feedable(
            data, batch_size, training=True, **kwargs
        )
        logs = []
        for batch_data, batch_label in zip(batch_data_list, batch_label_list):
            self.zero_grad()
            batch_pred = self.forward(batch_data)
            loss, log = self.loss_fn(batch_pred, batch_label)

            logs.append(log)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            optimizer.step()

        return summarize_logs(logs)

    def predict(self, test_data, **kwargs):
        self.eval()
        batch_data_list, _ = self.batch_sampler.convert_to_feedable(
            test_data, training=False, **kwargs
        )
        preds = [self.forward(batch_data).cpu().data.numpy() for batch_data in batch_data_list]
        return self.batch_sampler.reassemble(preds, test_data)
