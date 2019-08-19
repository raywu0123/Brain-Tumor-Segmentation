import torch

from .radam import RAdam
from utils import match_kwargs


class OptimizerFactory:

    custom_optimizers = {
        'RAdam': RAdam,
    }

    def __call__(
            self,
            model_parameters,
            dataset_size,
            optimizer_type='Adam',
            scheduler_type='MultiStepLR',
            epoch_milestones=(50, 70),
            **kwargs,
    ):
        if optimizer_type in self.custom_optimizers.keys():
            opt_constructor = self.custom_optimizers[optimizer_type]
        else:
            opt_constructor = eval(f'torch.optim.{optimizer_type}')

        scheduler_constructor = eval(f'torch.optim.lr_scheduler.{scheduler_type}')

        step_milestones = [n_epoch * dataset_size for n_epoch in epoch_milestones]
        optimizer = opt_constructor(
            model_parameters,
            **match_kwargs(opt_constructor, **kwargs),
        )
        scheduler = scheduler_constructor(
            optimizer,
            **match_kwargs(scheduler_constructor, milestones=step_milestones, **kwargs),
        )
        return optimizer, scheduler
