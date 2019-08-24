import torch
import numpy as np

from models.utils import get_tensor_from_array
from utils import epsilon
from .dice import generalized_soft_dice_score
from .utils import GetClassWeights


def ce_minus_log_dice(pred: torch.Tensor, tar: np.array):
    crossentropy_loss, log_1 = weighted_cross_entropy(pred, tar)
    dice_score, log_2 = generalized_soft_dice_score(pred, tar)
    total_loss = crossentropy_loss - torch.log(dice_score)
    return total_loss, {**log_1, **log_2}


def weighted_cross_entropy(output: torch.Tensor, target: np.array):
    assert(output.shape == target.shape)

    weights = GetClassWeights()(target)
    weights = get_tensor_from_array(weights)
    target = get_tensor_from_array(target)

    target = target.transpose(1, -1)
    output = output.transpose(1, -1)
    loss = target * weights * torch.log(output + epsilon)
    loss = -torch.mean(torch.sum(loss, dim=-1))
    return loss, {'crossentropy_loss': loss.item()}
