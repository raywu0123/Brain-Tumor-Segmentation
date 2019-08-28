from abc import ABC, abstractmethod

import torch
from torch import nn

from .batch_samplers import BatchSamplerHub
from .utils import summarize_logs
from .loss_functions import loss_function_hub


class ModelBase(ABC):

    @abstractmethod
    def fit_generator(self, training_data_generator, optimizer, **kwargs):
        pass

    @abstractmethod
    def predict(self, test_data, **kwargs):
        pass


class PytorchModelBase(ModelBase, nn.Module):

    def __init__(
            self,
            batch_sampler_id: str,
            loss_function_id: str,
            data_format: dict,
            clip_grad: float,
            optim_batch_steps: int
    ):
        nn.Module.__init__(self)
        self.loss_fn = loss_function_hub[loss_function_id]
        self.batch_sampler_constructor = BatchSamplerHub[batch_sampler_id]
        self.batch_sampler = self.batch_sampler_constructor(
            data_format=data_format
        )
        self.clip_grad = clip_grad
        self.batch_step_num = 0  # keeps count of how many batches processed
        self.optim_batch_steps = optim_batch_steps  # optimizer steps after this many steps

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

        self.zero_grad()
        for batch_data, batch_label in zip(batch_data_list, batch_label_list):
            batch_pred = self.forward(batch_data)
            loss, log = self.loss_fn(batch_pred, batch_label)
            loss /= self.optim_batch_steps
            logs.append(log)
            loss.backward()

            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad)

            self.batch_step_num += 1
            if self.batch_step_num % self.optim_batch_steps == 0:
                optimizer.step()
                self.zero_grad()

        return summarize_logs(logs)

    def predict(self, test_data, **kwargs):
        self.eval()
        with torch.no_grad():
            batch_data_list, _ = self.batch_sampler.convert_to_feedable(
                test_data, training=False, **kwargs
            )
            preds = [
                nn.functional.softmax(self.forward(batch_data), dim=1).cpu().data.numpy()
                for batch_data in batch_data_list
            ]
        return self.batch_sampler.reassemble(preds, test_data)
