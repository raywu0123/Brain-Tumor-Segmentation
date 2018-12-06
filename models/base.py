from abc import ABC, abstractmethod

import torch
from torch import nn

from .batch_samplers.base import BatchSamplerBase
from .utils import summarize_logs


class ModelBase(ABC):

    def __init__(self, batch_sampler: BatchSamplerBase):
        self.batch_sampler = batch_sampler

    @abstractmethod
    def fit_generator(self, training_data_generator, optimizer, **kwargs):
        pass

    @abstractmethod
    def predict(self, test_data, **kwargs):
        pass


class PytorchModelBase(ModelBase, nn.Module):

    def __init__(self, batch_sampler: BatchSamplerBase, loss_fn):
        nn.Module.__init__(self)
        ModelBase.__init__(self, batch_sampler=batch_sampler)
        self.loss_fn = loss_fn

    def fit_generator(self, training_data_generator, optimizer, **kwargs):
        self.train()
        data = training_data_generator(batch_size=1)
        batch_data_list, batch_label_list = self.batch_sampler.convert_to_feedable(
            data, training=True, **kwargs
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
