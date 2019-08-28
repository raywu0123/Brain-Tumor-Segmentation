import inspect

from keras.utils import to_categorical
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


def to_one_hot_label(label, class_num):
    categorical_label = to_categorical(label, class_num, dtype=np.bool)
    categorical_label = np.moveaxis(categorical_label, -1, 0)
    return categorical_label


def strip_file_extension(file_path):
    return file_path.strip('.nii.gz').strip('.npy')
