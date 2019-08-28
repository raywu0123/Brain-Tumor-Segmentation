import torch
import numpy as np

from ..utils import get_tensor_from_array
from utils import epsilon
from .utils import GetClassWeights


def naive_soft_dice_score(pred: torch.Tensor, tar: np.array):
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


def generalized_soft_dice_score(pred: torch.Tensor, tar: np.array):
    if not pred.shape == tar.shape:
        raise ValueError(f'Shape mismatch in pred and tar, got {pred.shape} and {tar.shape}')
    if not pred.shape[1] > 1:
        raise ValueError(f'Number of channels should be greater than 1, '
                         f'got data with shape {pred.shape}')

    class_weights = GetClassWeights()(tar) ** 2
    class_weights = get_tensor_from_array(class_weights)
    tar = get_tensor_from_array(tar)

    m1 = pred.view(pred.shape[0], pred.shape[1], -1)
    m2 = tar.view(tar.shape[0], tar.shape[1], -1)
    intersection = m1 * m2

    m1 = torch.sum(m1 ** 2, dim=2)
    m2 = torch.sum(m2 ** 2, dim=2)
    intersection = torch.sum(intersection, dim=2)

    num = torch.sum(2. * intersection * class_weights + epsilon, dim=-1)
    denom = torch.sum((m1 + m2) * class_weights + epsilon, dim=-1)

    dice_score = torch.mean(num / denom)
    return dice_score, {'soft_dice': dice_score.item()}


def my_soft_dice_score(pred: torch.Tensor, tar: np.array):
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


dice_score_hub = {
    'naive': naive_soft_dice_score,
    'generalized': generalized_soft_dice_score,
    'my': my_soft_dice_score,
}
