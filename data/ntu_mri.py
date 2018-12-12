import os

import nibabel as nib
import numpy as np
np.random.seed = 0

from .base import DataGeneratorBase
from .data_provider_base import DataProviderBase
from .utils import to_one_hot_label

from dotenv import load_dotenv

load_dotenv('./.env')

NTU_MRI_DIR = os.environ.get('NTU_MRI_DIR')
NTU_MOCK_TEST_DIR = os.environ.get('NTU_MOCK_TEST_DIR')
NTU_TEST_DIR = os.environ.get('NTU_TEST_DIR')


class NtuMriDataProvider(DataProviderBase):

    def __init__(self, args):
        self.data_dir = self._get_dir(args)
        self.image_path = os.path.join(self.data_dir, 'image')
        self.label_path = os.path.join(self.data_dir, 'label')
        self.all_ids = os.listdir(self.image_path)
        self.train_ids = self.all_ids[: -len(self.all_ids) // 10]
        self.test_ids = self.all_ids[-len(self.all_ids) // 10:]

    def _get_raw_data_generator(self, data_ids, **kwargs):
        return NtuDataGenerator(data_ids, self.data_format, data_dir=self.data_dir, **kwargs)

    def _get_dir(self, args):
        if 'mri' in args:
            return NTU_MRI_DIR
        if 'mocktest' in args:
            return NTU_MOCK_TEST_DIR
        if 'test' in args:
            return NTU_TEST_DIR
        else:
            raise KeyError('illegal args to ntu mri data provider.')

    @property
    def data_format(self):
        return {
            "channels": 1,
            "depth": 200,
            "height": 200,
            "width": 200,
            "class_num": 2,
        }


class NtuDataGenerator(DataGeneratorBase):

    def __init__(self, data_ids, data_format, data_dir, random=True):
        self.data_dir = data_dir
        self.data_ids = data_ids

        self._data_format = data_format
        self.random = random
        self.current_index = 0

        self.image_path = os.path.join(data_dir, 'image')
        self.label_path = os.path.join(data_dir, 'label')

    def __len__(self):
        return len(self.data_ids)

    def __call__(self, batch_size):
        if self.random:
            selected_data_ids = np.random.choice(self.data_ids, batch_size)
        else:
            selected_data_ids = self.data_ids[self.current_index: self.current_index + batch_size]
            self.current_index += batch_size
        return self._get_data(selected_data_ids)

    def _get_data(self, data_ids):
        batch_volume = np.empty((
            len(data_ids),
            self.data_format['channels'],
            self.data_format['depth'],
            self.data_format['height'],
            self.data_format['width'],
        ))
        batch_label = np.empty((
            len(data_ids),
            self.data_format['class_num'],
            self.data_format['depth'],
            self.data_format['height'],
            self.data_format['width'],
        ))

        for idx, data_id in enumerate(data_ids):
            batch_volume[idx], batch_label[idx] = self._get_image_and_label(data_id)
        return {'volume': batch_volume, 'label': batch_label}

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
            label = to_one_hot_label(label, self.data_format['class_num'])
        else:
            label = None

        return image, label
