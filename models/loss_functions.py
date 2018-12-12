import torch
import numpy as np

from .utils import get_tensor_from_array

epsilon = 1e-8


def ce_minus_log_dice(pred: torch.Tensor, tar: np.array):

    channel_num = tar.shape[1]
    temp = np.swapaxes(tar, 0, 1).reshape(channel_num, -1)
    class_weights = np.divide(
        1., np.mean(temp, axis=(1)),
        out=np.ones(tar.shape[1]),
        where=np.mean(temp, axis=(1)) != 0,
    )

    tar = get_tensor_from_array(tar)
    crossentropy_loss, log_1 = weighted_cross_entropy(pred, tar, weights=class_weights)
    dice_score, log_2 = soft_dice_score(pred, tar)
    total_loss = crossentropy_loss - torch.log(dice_score)
    return total_loss, {**log_1, **log_2}


def weighted_cross_entropy(output: torch.Tensor, target: torch.Tensor, weights=None):
    assert(output.shape == target.shape)
    if weights is None:
        weights = (1,) * output.shape[1]
    else:
        assert(len(weights) == output.shape[1])

    weights = torch.Tensor(weights)
    if torch.cuda.is_available():
        weights = weights.cuda()

    target = target.transpose(1, -1)
    output = output.transpose(1, -1)
    loss = target * weights * torch.log(output + epsilon)
    loss = -torch.mean(torch.sum(loss, dim=-1))
    return loss, {'crossentropy_loss': loss.item()}


def soft_dice_score(pred: torch.Tensor, tar: torch.Tensor):
    if not pred.shape == tar.shape:
        raise ValueError(f'Shape mismatch in pred and tar, got {pred.shape} and {tar.shape}')
    if not pred.shape[1] > 1:
        raise ValueError(f'Number of channels should be greater than 1, '
                         f'got data with shape {pred.shape}')
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
