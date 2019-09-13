import os

import nibabel as nib
import numpy as np
np.random.seed = 0

from .base import DataGeneratorBase
from .data_provider_base import DataProviderBase

from dotenv import load_dotenv

load_dotenv('./.env')

NTU_CT_DIR = os.environ.get('NTU_CT_DIR')


class NtuCtDataProvider(DataProviderBase):

    _data_format = {
        "channels": 1,
        "depth": 150,
        "height": 512,
        "width": 512,
        "class_num": 9,
    }

    def __init__(self, args):
        self.all_ids = os.listdir(NTU_CT_DIR)
        self.train_ids = self.all_ids[: -len(self.all_ids) // 10]
        self.test_ids = self.all_ids[-len(self.all_ids) // 10:]

    def _get_raw_data_generator(self, data_ids, **kwargs):
        return NtuCtDataGenerator(data_ids, self.data_format, **kwargs)

    @property
    def data_format(self) -> dict:
        return self._data_format


class NtuCtDataGenerator(DataGeneratorBase):

    def __init__(self, data_ids, data_format, random=True, **kwargs):
        super().__init__(data_ids, data_format, random)
        self.data_dir = NTU_CT_DIR

    def _get_data(self, data_ids):
        batch_volume = np.zeros((
            len(data_ids),
            self.data_format['channels'],
            self.data_format['depth'],
            self.data_format['height'],
            self.data_format['width'],
        ))
        batch_label = np.zeros((
            len(data_ids),
            self.data_format['depth'],
            self.data_format['height'],
            self.data_format['width'],
        ), dtype=np.uint8)

        affines = []
        for idx, data_id in enumerate(data_ids):
            volume, label, affine = self._preload_get_image_and_label(data_id)
            batch_volume[idx, :, :volume.shape[-3], :volume.shape[-2], :volume.shape[-1]] = \
                volume[
                    :self.data_format['depth'],
                    :self.data_format['height'],
                    :self.data_format['width']]
            batch_label[idx, :volume.shape[-3], :volume.shape[-2], :volume.shape[-1]] = \
                label[
                    :self.data_format['depth'],
                    :self.data_format['height'],
                    :self.data_format['width']]
            affines.append(affine)

        return {
            'volume': batch_volume,
            'label': batch_label,
            'data_ids': data_ids,
            'affines': affines,
        }

    def _preload_get_image_and_label(self, data_id):
        # Dims: (N, C, D, H, W)
        img_path = os.path.join(self.data_dir, f"{data_id}/data.nii.gz")
        image_obj = nib.load(img_path)
        affine = image_obj.affine
        image = image_obj.get_fdata()
        image = np.clip(image, a_min=-1000., a_max=None)
        image = np.transpose(image, (2, 0, 1))
        label_path = os.path.join(self.data_dir, f"{data_id}/label.nii.gz")

        if os.path.exists(label_path):
            label = nib.load(label_path).get_fdata()
            label = np.transpose(label, (2, 0, 1))
        else:
            label = None
        return image, label, affine
