import os

import SimpleITK as sitk
import numpy as np
np.random.seed = 0

from .base import DataGeneratorFactoryBase, DataGeneratorBase
from .utils import to_one_hot_label
from .wrappers import AsyncDataGeneratorWrapper
from utils import BRATSMetricClass

modal_bases = ['Flair.']
label_base = 'OT.'
data_extension = '.mha'


class Brats2015DataGeneratorFactory(DataGeneratorFactoryBase):

    def __init__(self, data_dirs):
        self._metric = BRATSMetricClass

        self.data_dirs = data_dirs

        self.all_ids = self._get_all_ids()

        self.train_ids = self.all_ids[: -len(self.all_ids) // 10]
        self.test_ids = self.all_ids[-len(self.all_ids) // 10:]
        print(f'training on {len(self.train_ids)} samples, '
              f'validating on {len(self.test_ids)} samples')

    def _get_all_ids(self):
        all_ids = []
        for data_dir in self.data_dirs:
            folder_names = os.listdir(data_dir)
            folder_dirs = [os.path.join(data_dir, foldername) for foldername in folder_names]
            all_ids.extend(folder_dirs)
        return all_ids

    def _get_data_generator(self, data_ids, **kwargs):
        data_generator = Brats2015DataGenerator(data_ids, self.data_format, **kwargs)
        AsyncDataGeneratorWrapper(data_generator)
        return data_generator

    def get_testing_data_generator(self, **kwargs):
        return self._get_data_generator(self.test_ids, **kwargs)

    def get_training_data_generator(self, **kwargs):
        return self._get_data_generator(self.train_ids, **kwargs)

    @property
    def data_format(self):
        return {
            "channels": len(modal_bases),
            "depth": 155,
            "height": 240,
            "width": 240,
            "class_num": 5,
        }


class Brats2015DataGenerator(DataGeneratorBase):

    def __init__(self, data_ids, data_format, random=True):
        self.data_ids = data_ids
        self.data_format = data_format
        self.random = random
        self.current_index = 0

    def __len__(self):
        return len(self.data_ids)

    def __call__(self, batch_size):
        if self.random:
            selected_data_ids = np.random.choice(self.data_ids, batch_size)
        else:
            selected_data_ids = self.data_ids[self.current_index: self.current_index + batch_size]
            self.current_index += batch_size
        return self._get_data(selected_data_ids)

    def _get_image_and_label(self, data_id):
        image = [self._get_image_from_folder(data_id, base) for base in modal_bases]
        image = np.asarray(image)
        label = self._get_image_from_folder(data_id, label_base)
        label = to_one_hot_label(label, self.data_format['class_num'])
        return image, label

    @staticmethod
    def _get_image_from_folder(folder_dir, match_string):
        modal_folder = [f for f in os.listdir(folder_dir) if match_string in f]
        assert(len(modal_folder) == 1)
        modal_folder_dir = os.path.join(folder_dir, modal_folder[0])

        data_filename = [f for f in os.listdir(modal_folder_dir) if data_extension in f]
        assert(len(data_filename) == 1)
        data_dir = os.path.join(modal_folder_dir, data_filename[0])

        image = sitk.ReadImage(data_dir)
        image_array = sitk.GetArrayFromImage(image)
        return image_array

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
