import torch
import numpy as np


def normalize_batch_image(batch_image):
    assert(batch_image.ndim == 4)
    std = np.std(batch_image, axis=(1, 2, 3), keepdims=True)
    std_is_zero = std == 0
    batch_image = (batch_image - np.mean(batch_image, axis=(1, 2, 3), keepdims=True)) \
        / (std + std_is_zero.astype(float))
    return batch_image


def co_shuffle(batch_data, batch_label):
    assert(len(batch_data) == len(batch_label))
    p = np.random.permutation(len(batch_data))
    batch_data = batch_data[p]
    batch_label = batch_label[p]
    return batch_data, batch_label


def get_tensor_from_array(array):
    tensor = torch.Tensor(array)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def summarize_logs(logs: [dict]) -> dict:
    summary = {}
    if len(logs) == 0:
        return summary

    for key in logs[0].keys():
        summary[key] = np.mean([d[key] for d in logs])

    summary['data_count'] = len(logs)
    return summary
