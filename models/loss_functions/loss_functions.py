import torch
import numpy as np

from models.utils import get_tensor_from_array
from utils import epsilon


def ce_minus_log_dice(pred: torch.Tensor, tar: np.array):
    crossentropy_loss, log_1 = weighted_cross_entropy(pred, tar)
    dice_score, log_2 = soft_dice_score(pred, tar)
    total_loss = crossentropy_loss - torch.log(dice_score)
    return total_loss, {**log_1, **log_2}


def minus_dice(pred: torch.Tensor, tar: np.array):
    dice_score, log = soft_dice_score(pred, tar)
    return -dice_score, log


def weighted_cross_entropy(output: torch.Tensor, target: np.array):
    assert(output.shape == target.shape)

    channel_num = target.shape[1]
    temp = np.swapaxes(target, 0, 1).reshape(channel_num, -1)
    weights = np.divide(
        1., np.mean(temp, axis=1),
        out=np.ones(target.shape[1]),
        where=np.mean(temp, axis=1) != 0,
    )
    weights *= channel_num / np.sum(weights)

    weights = get_tensor_from_array(weights)
    target = get_tensor_from_array(target)

    target = target.transpose(1, -1)
    output = output.transpose(1, -1)
    loss = target * weights * torch.log(output + epsilon)
    loss = -torch.mean(torch.sum(loss, dim=-1))
    return loss, {'crossentropy_loss': loss.item()}


def soft_dice_score(pred: torch.Tensor, tar: np.array):
    if not pred.shape == tar.shape:
        raise ValueError(f'Shape mismatch in pred and tar, got {pred.shape} and {tar.shape}')
    if not pred.shape[1] > 1:
        raise ValueError(f'Number of channels should be greater than 1, '
                         f'got data with shape {pred.shape}')
    tar = get_tensor_from_array(tar)

    # Strip background
    pred = pred[:, 1:]
    tar = tar[:, 1:]

    m1 = pred.view(pred.shape[0], pred.shape[1], -1)
    m2 = tar.view(tar.shape[0], tar.shape[1], -1)
    intersection = m1 * m2

    m1 = torch.sum(m1 ** 2, dim=2)
    m2 = torch.sum(m2 ** 2, dim=2)
    intersection = torch.sum(intersection, dim=2)

    dice_score = torch.mean((2. * intersection + epsilon) / (m1 + m2 + epsilon))
    return dice_score, {'soft_dice': dice_score.item()}


def _soft_dice_score_deprecated(pred: torch.Tensor, tar: np.array):
    if not pred.shape == tar.shape:
        raise ValueError(f'Shape mismatch in pred and tar, got {pred.shape} and {tar.shape}')
    if not pred.shape[1] > 1:
        raise ValueError(f'Number of channels should be greater than 1, '
                         f'got data with shape {pred.shape}')
    tar = get_tensor_from_array(tar)

    # Strip background
    pred = pred[:, 1:]
    tar = tar[:, 1:]

    m1 = pred.view(pred.shape[0], pred.shape[1], -1)
    m2 = tar.view(tar.shape[0], tar.shape[1], -1)
    intersection = m1 * m2

    m1 = torch.sum(m1 ** 2)
    m2 = torch.sum(m2 ** 2)
    intersection = torch.sum(intersection)

    dice_score = torch.mean((2. * intersection + epsilon) / (m1 + m2 + epsilon))
    return dice_score, {'soft_dice': dice_score.item()}
