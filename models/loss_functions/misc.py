import torch
from torch import nn
import numpy as np

from models.utils import get_tensor_from_array
from .dice import dice_score_hub
from .utils import GetClassWeights
from utils import to_one_hot_label


def ce_minus_log_dice(logits: torch.Tensor, tar: np.array, dice_type: str = 'my'):
    crossentropy_loss, log_1 = weighted_cross_entropy(logits, tar)

    dice_fn = dice_score_hub[dice_type]
    onehot_tar = to_one_hot_label(tar, class_num=logits.shape[1])
    dice_score, log_2 = dice_fn(logits, onehot_tar)

    total_loss = crossentropy_loss - torch.log(dice_score)
    return total_loss, {**log_1, **log_2}


def weighted_cross_entropy(logits: torch.Tensor, target: np.array):
    weights = GetClassWeights()(target, class_num=logits.shape[1])
    weights = get_tensor_from_array(weights)
    target = get_tensor_from_array(target).long()
    loss = nn.CrossEntropyLoss(weight=weights)(logits, target)
    return loss, {'crossentropy_loss': loss.item()}


def minus_dice(logits: torch.Tensor, tar: np.array, dice_type: str = 'my'):
    onehot_tar = to_one_hot_label(tar, class_num=logits.shape[1])
    dice_fn = dice_score_hub[dice_type]
    dice_score, log = dice_fn(logits, onehot_tar)
    return -dice_score, log
