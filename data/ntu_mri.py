import os
from functools import partial

import nibabel as nib
from tqdm import tqdm
import numpy as np
np.random.seed = 0

from .base import DataGeneratorFactoryBase
from .utils import to_one_hot_label


img_channels = 1
img_depth = 200
img_height = img_width = 200
metadata_dim = 0
class_num = 2


class NtuMriDataGeneratorFactory(DataGeneratorFactoryBase):
    def __init__(self, data_dir):
        self.DATA_DIR = data_dir
        self.image_path = os.path.join(data_dir, 'image')
        self.label_path = os.path.join(data_dir, 'label')
        self.description = 'NTU_MRI'
        self.all_ids = os.listdir(self.image_path)
        self.train_ids = self.all_ids[: -len(self.all_ids) // 10]
        self.test_ids = self.all_ids[-len(self.all_ids) // 10:]

    def _data_generator(self, data_ids, batch_size):
        selected_ids = np.random.choice(data_ids, batch_size)
        return self._get_data(selected_ids)

    def _get_data(self, data_ids, verbose=False):
        batch_volume = np.empty((
            len(data_ids),
            img_channels,
            img_depth,
            img_height,
            img_width,
        ))
        batch_label = np.empty((
            len(data_ids),
            class_num,
            img_depth,
            img_height,
            img_width,
        ))

        iterator = data_ids
        if verbose:
            print('Loading data...')
            iterator = tqdm(data_ids)

        for idx, data_id in enumerate(iterator):
            batch_volume[idx], batch_label[idx] = self._get_image_and_label(data_id)
        return {'volume': batch_volume, 'metadata': None, 'label': batch_label}

    def _get_image_and_label(self, data_id):
        # Dims: (N, C, D, H, W)
        img_path = os.path.join(self.image_path, data_id)
        image_obj = nib.load(img_path)
        image = image_obj.get_fdata()
        image = np.transpose(image, (2, 0, 1))

        label_path = os.path.join(self.label_path, data_id)
        if os.path.exists(label_path):
            label = nib.load(label_path).get_fdata()
            label = np.transpose(label, (2, 0, 1))
            label = to_one_hot_label(label, class_num)
        else:
            label = None

        return image, label

    def get_testing_data_generator(self, **kwargs):
        return partial(self._data_generator, self.test_ids)

    def get_training_data_generator(self, **kwargs):
        return partial(self._data_generator, self.train_ids)

    @property
    def data_format(self):
        data_format = {
            "channels": img_channels,
            "depth": img_depth,
            "height": img_height,
            "width": img_width,
            "metadata_dim": metadata_dim,
            "class_num": class_num,
        }
        return data_format
