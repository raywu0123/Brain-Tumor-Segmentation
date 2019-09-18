import inspect

import numpy as np


epsilon = 1e-8


def highlight_print(msg, highlight_len=30):
    print('#' * highlight_len)
    print(msg)
    print('#' * highlight_len)


def get_2d_from_3d(batch_volume):
    assert(batch_volume.ndim == 4 or batch_volume.ndim == 5)
    if batch_volume.ndim == 5:
        batch_volume = np.transpose(batch_volume, (0, 2, 1, 3, 4))
        batch_image = batch_volume.reshape(-1, *batch_volume.shape[-3:])
    else:
        batch_image = batch_volume.reshape(-1, *batch_volume.shape[-2:])
    return batch_image


def get_3d_from_2d(batch_image, data_depth):
    assert(batch_image.ndim == 3 or batch_image.ndim == 4)
    if batch_image.ndim == 3:
        batch_volume = batch_image.reshape(-1, data_depth, *batch_image.shape[-2:])
    else:
        batch_volume = batch_image.reshape(-1, data_depth, *batch_image.shape[-3:])
        batch_volume = batch_volume.transpose([0, 2, 1, 3, 4])
    return batch_volume


def parse_exp_id(exp_id_string):
    splits = exp_id_string.split('_on_')
    model_id = splits[0]
    data_id = '_'.join(splits[-1].split('_')[:-1])
    time_stamp = splits[-1].split('_')[-1]
    return model_id, data_id, time_stamp


def match_kwargs(func, **kwargs):
    ret = {}
    for key, value in kwargs.items():
        if key in inspect.getfullargspec(func)[0]:
            ret[key] = value
    return ret


def to_one_hot_label(y, class_num=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not class_num:
        class_num = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, class_num), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (class_num,)
    categorical = np.reshape(categorical, output_shape)
    categorical = np.moveaxis(categorical, -1, 1)
    return categorical


def strip_file_extension(file_path):
    return file_path.strip('.nii.gz').strip('.npy')
