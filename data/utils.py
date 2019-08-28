from keras.utils import to_categorical
import numpy as np


def to_one_hot_label(label, class_num):
    categorical_label = to_categorical(label, class_num, dtype=np.bool)
    categorical_label = np.moveaxis(categorical_label, -1, 0)
    return categorical_label


def strip_file_extension(file_path):
    return file_path.strip('.nii.gz').strip('.npy')
