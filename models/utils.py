import torch
import numpy as np


def normalize_batch_image(batch_image):
    assert(batch_image.ndim == 4 or batch_image.ndim == 5)
    axis = tuple(range(1, batch_image.ndim))
    std = np.std(batch_image, axis=axis, keepdims=True)
    mean = np.mean(batch_image, axis=axis, keepdims=True)
    std_is_zero = std == 0
    batch_image = (batch_image - mean) / (std + std_is_zero.astype(float))
    return batch_image


def co_shuffle(*args):
    for item in args[1:]:
        assert len(item) == len(args[0])
    p = np.random.permutation(len(args[0]))
    return (item[p] for item in args)


def get_tensor_from_array(array: np.array):
    if array.dtype == np.bool:
        array = array.astype(np.uint8)

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
