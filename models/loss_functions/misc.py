import torch
from torch import nn
import numpy as np

from models.utils import get_tensor_from_array
from .dice import dice_score_hub
from .utils import GetClassWeights
from utils import to_one_hot_label


def ce_minus_log_dice(logits: torch.Tensor, tar: np.array, dice_type: str = 'my', **kwargs):
    crossentropy_loss, log_1 = weighted_cross_entropy(logits, tar)

    dice_fn = dice_score_hub[dice_type]
    onehot_tar = to_one_hot_label(tar, class_num=logits.shape[1])
    dice_score, log_2 = dice_fn(logits, onehot_tar)

    total_loss = crossentropy_loss - torch.log(dice_score)
    return total_loss, {**log_1, **log_2}


def weighted_cross_entropy(logits: torch.Tensor, target: np.array, **kwargs):
    weights = GetClassWeights()(target, class_num=logits.shape[1])
    weights = get_tensor_from_array(weights)
    target = get_tensor_from_array(target).long()
    loss = nn.CrossEntropyLoss(weight=weights)(logits, target)
    return loss, {'crossentropy_loss': loss.item()}


def minus_dice(logits: torch.Tensor, tar: np.array, dice_type: str = 'my', **kwargs):
    onehot_tar = to_one_hot_label(tar, class_num=logits.shape[1])
    dice_fn = dice_score_hub[dice_type]
    dice_score, log = dice_fn(logits, onehot_tar)
    return -dice_score, log


class MultiTaskLossWrapper(nn.Module):

    def __init__(self, loss_fn, class_num):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(size=[class_num]))
        self.loss_fn = loss_fn

    def __call__(self, pred, tar, data_idx):
        original_loss, log = self.loss_fn(pred, tar)
        log_var = self.log_vars[data_idx]
        new_loss = original_loss / (2 * torch.exp(log_var)) + 0.5 * log_var

        log_var_log = {
            f'log_var_{idx}': log_var.item()
            for idx, log_var in enumerate(self.log_vars)
        }
        return new_loss, {'weighted_loss': new_loss.item(), **log_var_log, **log}
