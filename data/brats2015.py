from functools import partial
import os

from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
np.random.seed = 0

from .base import DataGeneratorFactoryBase
from .utils import to_one_hot_label
from utils import BRATSMetricClass

modal_bases = ['Flair.']
label_base = 'OT.'
data_extension = '.mha'


class Brats2015DataGeneratorFactory(DataGeneratorFactoryBase):
    def __init__(self, data_dirs):
        self._metric = BRATSMetricClass

        self.img_channels = len(modal_bases)
        self.img_depth = 155
        self.img_height = self.img_width = 240
        self.metadata_dim = 0
        self.class_num = 5

        self.description = 'BRATS2015'
        self.DATA_DIRS = data_dirs

        self.all_ids = []
        for data_dir in data_dirs:
            folder_names = os.listdir(data_dir)
            folder_dirs = [os.path.join(data_dir, foldername) for foldername in folder_names]
            self.all_ids.extend(folder_dirs)

        self.train_ids = self.all_ids[: -len(self.all_ids) // 10]
        self.test_ids = self.all_ids[-len(self.all_ids) // 10:]
        print(f'training on {len(self.train_ids)} samples, '
              f'validating on {len(self.test_ids)} samples')

    def _get_image_and_label(self, data_id):
        image = [self._get_image_from_folder(data_id, base) for base in modal_bases]
        image = np.asarray(image)
        label = self._get_image_from_folder(data_id, label_base)
        label = to_one_hot_label(label, self.class_num)
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

    def _data_generator(self, data_ids, batch_size):
        selected_ids = np.random.choice(data_ids, batch_size)
        return self._get_data(selected_ids)

    def get_testing_data_generator(self, **kwargs):
        return partial(self._data_generator, self.test_ids)

    def get_training_data_generator(self, **kwargs):
        return partial(self._data_generator, self.train_ids)

    def _get_data(self, data_ids, verbose=False):
        batch_volume = np.empty((
            len(data_ids),
            self.img_channels,
            self.img_depth,
            self.img_height,
            self.img_width,
        ))
        batch_label = np.empty((
            len(data_ids),
            self.class_num,
            self.img_depth,
            self.img_height,
            self.img_width,
        ))

        iterator = data_ids
        if verbose:
            print('Loading data...')
            iterator = tqdm(data_ids)

        for idx, data_id in enumerate(iterator):
            batch_volume[idx], batch_label[idx] = self._get_image_and_label(data_id)
        return {'volume': batch_volume, 'metadata': None, 'label': batch_label}

    @property
    def data_format(self):
        data_format = {
            "channels": self.img_channels,
            "depth": self.img_depth,
            "height": self.img_height,
            "width": self.img_width,
            "metadata_dim": self.metadata_dim,
            "class_num": self.class_num,
        }
        return data_format
