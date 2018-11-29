from functools import partial
import os

from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
np.random.seed = 0

from .base import DataInterface
from preprocess_tools.image_utils import save_array_to_nii
from .utils import to_one_hot_label

from dotenv import load_dotenv
load_dotenv('./.env')
BRATS2015_DIR = os.environ.get('BRATS2015_DIR')
BRATS2015_HGG_DIR = os.path.join(BRATS2015_DIR, './HGG')
BRATS2015_LGG_DIR = os.path.join(BRATS2015_DIR, './LGG')

#   modal_bases = ['Flair.', 'T1.', 'T1c.', 'T2.']
label_base = 'OT.'
data_extension = '.mha'


def get_brats_modality(args):
    modal_bases = []
    if 'Flair' in args:
        modal_bases.append('Flair.')
    if 'T1' in args:
        modal_bases.append('T1.')
    if 'T1c' in args:
        modal_bases.append('T1c.')
    if 'T2c' in args:
        modal_bases.append('T2.')
    #   default
    if not modal_bases:
        modal_bases = ['Flair.', 'T1.', 'T1c.', 'T2.']
    return modal_bases


def get_brats_tumorType(args):
    data_dirs = []
    if 'hgg' in args:
        data_dirs.append(BRATS2015_HGG_DIR)
    if 'lgg' in args:
        data_dirs.append(BRATS2015_LGG_DIR)
    #   default
    if not data_dirs:
        data_dirs = [BRATS2015_HGG_DIR, BRATS2015_LGG_DIR]
    return data_dirs


class BRATS2015(DataInterface):
    def __init__(self, args):
        self.img_channels = 4
        self.img_depth = 155
        self.img_height = self.img_width = 240
        self.metadata_dim = 0
        self.class_num = 5

        self.description = 'BRATS2015'
        self.DATA_DIRS = get_brats_tumorType(args)
        self.modal_bases = get_brats_modality(args)

        self.all_ids = []
        for DATA_DIR in self.DATA_DIRS:
            folder_names = os.listdir(DATA_DIR)
            folder_dirs = [os.path.join(DATA_DIR, foldername) for foldername in folder_names]
            self.all_ids.extend(folder_dirs)

        self.train_ids = self.all_ids[: -len(self.all_ids) // 10]
        self.test_ids = self.all_ids[-len(self.all_ids) // 10:]
        print(f'training on {len(self.train_ids)} samples, '
              f'validating on {len(self.test_ids)} samples')

    def _get_image_and_label(self, data_id):
        image = [self._get_image_from_folder(data_id, base) for base in self.modal_bases]
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

    def _datagenerator(self, data_ids, batch_size):
        selected_ids = np.random.choice(data_ids, batch_size)
        return self._get_data(selected_ids)

    @property
    def testing_datagenerator(self):
        return partial(self._datagenerator, self.test_ids)

    @property
    def training_datagenerator(self):
        return partial(self._datagenerator, self.train_ids)

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

    def get_training_data(self):
        return self._get_data(self.train_ids, verbose=True)

    def get_testing_data(self):
        return self._get_data(self.test_ids, verbose=True)

    def get_all_data(self):
        return self._get_data(self.all_ids, verbose=True)

    def get_data_format(self):
        data_format = {
            "channels": self.img_channels,
            "depth": self.img_depth,
            "height": self.img_height,
            "width": self.img_width,
            "metadata_dim": self.metadata_dim,
            "class_num": self.class_num,
        }
        return data_format

    def save_result(self, np_array, save_path, data_id):
        save_array_to_nii(np_array, save_path, self.original_niis[data_id])
