import os
from functools import partial

# from bistiming import SimpleTimer
# import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
np.random.seed = 0

from .base import DataInterface


load_dotenv('./.env')
NTU_MRI_DIR = os.environ.get('NTU_MRI_DIR')


class NTU_MRI(DataInterface):
    def __init__(self):
        self.img_channels = 1
        self.img_depth = 200
        self.img_height = self.img_width = 200
        self.metadata_dim = 0

        self.description = 'NTU_MRI'
        self.image_path = os.path.join(NTU_MRI_DIR, 'image')
        self.label_path = os.path.join(NTU_MRI_DIR, 'label')

        self.all_ids = os.listdir(self.image_path)
        self.train_ids = self.all_ids[: -len(self.all_ids) // 10]
        self.test_ids = self.all_ids[-len(self.all_ids) // 10:]

    def _get_image_and_label(self, data_id):
        # Dims: (N, C, D, H, W)
        img_path = os.path.join(self.image_path, data_id)
        # image = nib.load(img_path).get_fdata()
        image = np.load(img_path)
        image = np.transpose(image, (2, 0, 1))

        label_path = os.path.join(self.label_path, data_id)
        if os.path.exists(label_path):
            # label = nib.load(label_path).get_fdata()
            label = np.load(label_path)
            label = np.transpose(label, (2, 0, 1))
        else:
            label = None
        return image, label

    def _data_generator(self, data_ids, batch_size):
        selected_ids = np.random.choice(data_ids, batch_size)
        return self._get_data(selected_ids)

    @property
    def testing_data_generator(self):
        return partial(self._data_generator, self.test_ids)

    @property
    def training_data_generator(self):
        return partial(self._data_generator, self.train_ids)

    def _get_data(self, data_ids):
        batch_volume = np.empty((
            len(data_ids),
            self.img_channels,
            self.img_depth,
            self.img_height,
            self.img_width,
        ))
        batch_label = np.empty_like(batch_volume)

        print('Loading data...')
        for idx, data_id in enumerate(tqdm(data_ids)):
            batch_volume[idx], batch_label[idx] = self._get_image_and_label(data_id)
        return {'volume': batch_volume, 'metadata': None, 'label': batch_label}

    def get_training_data(self):
        return self._get_data(self.train_ids)

    def get_testing_data(self):
        return self._get_data(self.test_ids)

    def get_all_data(self):
        return self._get_data(self.train_ids + self.test_ids)

    def get_data_format(self):
        data_format = {
            "channels": self.img_channels,
            "depth": self.img_depth,
            "height": self.img_height,
            "width": self.img_width,
            "metadata_dim": self.metadata_dim,
        }
        return data_format
