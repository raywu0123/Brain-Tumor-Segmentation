import os

import SimpleITK as sitk
import numpy as np
np.random.seed = 0

from .data_provider_base import DataProviderBase
from .base import DataGeneratorBase
from metrics import BRATSMetric, ClasswiseMetric

from dotenv import load_dotenv

label_base = 'OT.'
data_extension = '.mha'

load_dotenv('./.env')

BRATS2015_DIR = os.environ.get('BRATS2015_DIR')
BRATS2015_HGG_DIR = os.path.join(BRATS2015_DIR, './HGG')
BRATS2015_LGG_DIR = os.path.join(BRATS2015_DIR, './LGG')


class Brats2015DataProvider(DataProviderBase):

    def __init__(self, args):
        self.data_dirs = self._get_dirs(args)

        self.modal_bases = self._get_modal_bases(args)
        self.label_type, self._metric = self._get_label_type(args)

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

    def _get_raw_data_generator(self, data_ids, **kwargs):
        return Brats2015DataGenerator(
            data_ids, 
            self.data_format, 
            self.modal_bases, 
            self.label_type,
            **kwargs,
        )

    def _get_dirs(self, args):
        data_dirs = []
        if 'hgg' in args:
            data_dirs.append(BRATS2015_HGG_DIR)
        if 'lgg' in args:
            data_dirs.append(BRATS2015_LGG_DIR)
        #   default
        if not data_dirs:
            data_dirs = [BRATS2015_HGG_DIR, BRATS2015_LGG_DIR]
        return data_dirs

    def _get_modal_bases(self, args):
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

    def _get_label_type(self, args):
        label_type = ''
        for t in ['core', 'enhancing', 'complete']:
            if t in args:
                label_type = t
                break
        
        metric = BRATSMetric if not label_type else ClasswiseMetric
        return label_type, metric

    @property
    def data_format(self):
        return {
            "channels": len(self.modal_bases),
            "depth": 155,
            "height": 240,
            "width": 240,
            "class_num": 5 if not self.label_type else 2,
        }


class Brats2015DataGenerator(DataGeneratorBase):

    def __init__(self, data_ids, data_format, modal_bases, label_type, random=True, **kwargs):
        super().__init__(data_ids, data_format, random)
        self.modal_bases = modal_bases
        self.label_type = label_type

    def _get_image_and_label(self, data_id):
        image = [self._get_image_from_folder(data_id, base) for base in self.modal_bases]
        image = np.asarray(image)
        label = self._get_image_from_folder(data_id, label_base)
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
            self.data_format['depth'],
            self.data_format['height'],
            self.data_format['width'],
        ), dtype=np.uint8)

        for idx, data_id in enumerate(data_ids):
            batch_volume[idx], batch_label[idx] = self._get_image_and_label(data_id)

        if self.label_type == 'core':
            batch_label = np.logical_and(batch_label != 0, batch_label != 2).astype(np.uint8)
        elif self.label_type == 'enhancing':
            batch_label = (batch_label == 4).astype(np.uint8)
        elif self.label_type == 'complete':
            batch_label = (batch_label != 0).astype(np.uint8)

        return {'volume': batch_volume, 'label': batch_label, 'data_ids': data_ids}
