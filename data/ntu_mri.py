import os
from functools import partial

import nibabel as nib
from tqdm import tqdm
from dotenv import load_dotenv
import numpy as np
np.random.seed = 0
from torch.utils.data import DataLoader, Dataset

from .base import DataInterface
from preprocess_tools.image_utils import save_array_to_nii

load_dotenv('./.env')

img_channels = 1
img_depth = 200
img_height = img_width = 200
metadata_dim = 0


class NTU_MRI_LOADING_BASE:
    def __init__(self, DATA_DIR):
        self.DATA_DIR = DATA_DIR
        self.image_path = os.path.join(DATA_DIR, 'image')
        self.label_path = os.path.join(DATA_DIR, 'label')
        self.original_niis = {}

    def _get_data(self, data_ids, verbose=False):
        batch_volume = np.empty((
            len(data_ids),
            img_channels,
            img_depth,
            img_height,
            img_width,
        ))
        batch_label = np.empty_like(batch_volume)

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
        else:
            label = None

        self.original_niis[data_id] = image_obj
        return image, label


class NTU_MRI(DataInterface, NTU_MRI_LOADING_BASE):
    def __init__(self, DATA_DIR):
        super().__init__(DATA_DIR)
        self.description = 'NTU_MRI'
        self.all_ids = os.listdir(self.image_path)
        self.train_ids = self.all_ids[: -len(self.all_ids) // 10]
        self.test_ids = self.all_ids[-len(self.all_ids) // 10:]

    def _datagenerator(self, data_ids, batch_size):
        selected_ids = np.random.choice(data_ids, batch_size)
        return self._get_data(selected_ids)

    @property
    def testing_datagenerator(self):
        return partial(self._datagenerator, self.test_ids)

    @property
    def training_datagenerator(self):
        return partial(self._datagenerator, self.train_ids)

    def get_training_data(self):
        return self._get_data(self.train_ids, verbose=True)

    def get_testing_data(self):
        return self._get_data(self.test_ids, verbose=True)

    def get_all_data(self):
        return self._get_data(self.all_ids, verbose=True)

    def get_training_dataloader(self, batch_size, shuffle, num_workers):
        return DataLoader(
            NTU_MRI_DATASET(self.train_ids, self.DATA_DIR),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    def get_testing_dataloader(self, batch_size, shuffle, num_workers):
        return DataLoader(
            NTU_MRI_DATASET(self.test_ids, self.DATA_DIR),
            batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )

    def get_data_format(self):
        data_format = {
            "channels": img_channels,
            "depth": img_depth,
            "height": img_height,
            "width": img_width,
            "metadata_dim": metadata_dim,
        }
        return data_format

    def save_result(self, np_array, save_path, data_id):
        save_array_to_nii(np_array, save_path, self.original_niis[data_id])


class NTU_MRI_DATASET(Dataset, NTU_MRI_LOADING_BASE):
    def __init__(self, data_ids, DATA_DIR):
        super().__init__(DATA_DIR)
        self.data_ids = data_ids

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, idx):
        image, label = self._get_image_and_label(self.data_ids[idx])
        return {'volume': image, 'label': label, 'metadata': []}
