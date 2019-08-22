import torch
import numpy as np

from .utils import get_tensor_from_array

epsilon = 1e-8


def ce_minus_log_dice(pred: torch.Tensor, tar: np.array):
    crossentropy_loss, log_1 = weighted_cross_entropy(pred, tar)
    dice_score, log_2 = soft_dice_score(pred, tar)
    total_loss = crossentropy_loss - torch.log(dice_score)
    return total_loss, {**log_1, **log_2}


def ce_minus_log_dice_with_size_mismatch(output: torch.Tensor, target: np.array):
    s_d = target.shape[2] // 2 - output.shape[2] // 2
    s_h = target.shape[3] // 2 - output.shape[3] // 2
    s_w = target.shape[4] // 2 - output.shape[4] // 2
    e_d = s_d + output.shape[2]
    e_h = s_h + output.shape[3]
    e_w = s_w + output.shape[4]
    target = target[:, :, s_d:e_d, s_h:e_h, s_w:e_w]
    return ce_minus_log_dice(output, target)


def weighted_cross_entropy(output: torch.Tensor, target: np.array):
    assert(output.shape == target.shape)

    channel_num = target.shape[1]
    temp = np.swapaxes(target, 0, 1).reshape(channel_num, -1)
    weights = np.divide(
        1., np.mean(temp, axis=1),
        out=np.ones(target.shape[1]),
        where=np.mean(temp, axis=1) != 0,
    )

    weights = get_tensor_from_array(weights)
    target = get_tensor_from_array(target)

    target = target.transpose(1, -1)
    output = output.transpose(1, -1)
    loss = target * weights * torch.log(output + epsilon)
    loss = -torch.mean(torch.sum(loss, dim=-1))
    return loss, {'crossentropy_loss': loss.item()}


def weighted_cross_entropy_with_size_mismatch(output: torch.Tensor, target: np.array):
    s_d = target.shape[2] // 2 - output.shape[2] // 2
    s_h = target.shape[3] // 2 - output.shape[3] // 2
    s_w = target.shape[4] // 2 - output.shape[4] // 2
    e_d = s_d + output.shape[2]
    e_h = s_h + output.shape[3]
    e_w = s_w + output.shape[4]
    target = target[:, :, s_d:e_d, s_h:e_h, s_w:e_w]
    return weighted_cross_entropy(output, target)


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

    m1 = torch.sum(m1 ** 2)
    m2 = torch.sum(m2 ** 2)
    intersection = torch.sum(intersection)

    dice_score = torch.mean((2. * intersection + epsilon) / (m1 + m2 + epsilon))
    return dice_score, {'soft_dice': dice_score.item()}
