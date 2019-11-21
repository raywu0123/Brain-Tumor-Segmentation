from abc import ABC, abstractmethod

import numpy as np
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
            data_format: dict,
            forward_outcome_channels: int,
            head_outcome_channels: int,
            auxiliary_data_formats: list,
            loss_function_id: str = 'crossentropy',
            clip_grad: float = 0.,
            optim_batch_steps: int = 1,
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

        all_data_formats = [data_format] + auxiliary_data_formats
        data_channels = [
            _data_format['channels']
            for _data_format in all_data_formats
        ]
        class_nums = [
            _data_format['class_num']
            for _data_format in all_data_formats
        ]
        self.heads = self.build_heads(
            input_channels=data_channels,
            output_channel=head_outcome_channels,
        )
        self.tails = self.build_tails(
            input_channels=forward_outcome_channels,
            class_nums=class_nums,
        )

    def fit_generator(
            self,
            training_data_generator,
            aux_data_generators,
            optimizer,
            batch_size,
            **kwargs,
    ):
        """
        fit on generator for one single volume
        """
        self.train()
        all_data_generators = [training_data_generator] + aux_data_generators
        all_data = [
            data_generator(batch_size=1)
            for data_generator in all_data_generators
        ]
        # batch_size here stands for number of volumes
        # most devices can only hold one singe volume
        batch_data_list = []
        batch_label_list = []
        data_idx_list = []
        for data_idx, data in enumerate(all_data):
            data_list, label_list = self.batch_sampler.convert_to_feedable(
                data, batch_size, training=True, **kwargs
            )
            batch_data_list.extend(data_list)
            batch_label_list.extend(label_list)
            data_idx_list.extend([data_idx] * len(data_list))

        logs = [[] for _ in all_data]

        self.zero_grad()
        sample_batch_order = np.random.permutation(len(batch_data_list))
        params = self.parameters()
        for idx in sample_batch_order:
            batch_data, batch_label, data_idx = \
                batch_data_list[idx], batch_label_list[idx], data_idx_list[idx]
            batch_pred = self.forward_head(batch_data, data_idx)
            batch_pred = self.forward(batch_pred)
            batch_pred = self.tails[data_idx](batch_pred)
            loss, log = self.loss_fn(batch_pred, batch_label)
            loss /= self.optim_batch_steps
            logs[data_idx].append(log)
            loss.backward()

            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(params, self.clip_grad)

            self.batch_step_num += 1
            if self.batch_step_num % self.optim_batch_steps == 0:
                optimizer.step()
                self.zero_grad()

        logs = [summarize_logs(log) for log in logs]
        return logs[0], logs[1:]

    def predict(self, test_data, **kwargs):
        """
        currently only for main data, i.e. tail_id=0
        """
        self.eval()
        with torch.no_grad():
            batch_data_list, _ = self.batch_sampler.convert_to_feedable(
                test_data, training=False, **kwargs
            )
            preds = [
                nn.functional.softmax(
                    self.tails[0](self.forward(self.forward_head(batch_data, 0))),
                    dim=1
                ).cpu().data.numpy()
                for batch_data in batch_data_list
            ]
        return self.batch_sampler.reassemble(preds, test_data)

    @abstractmethod
    def forward_head(self, inp, data_idx: int):
        pass

    @abstractmethod
    def build_heads(self, input_channels: list, output_channel: int):
        pass

    @abstractmethod
    def build_tails(self, input_channels: int, class_nums: list):
        pass
