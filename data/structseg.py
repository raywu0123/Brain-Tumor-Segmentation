import os

from tqdm import tqdm
import nibabel as nib
import numpy as np
np.random.seed = 0

from .base import DataGeneratorBase
from .data_provider_base import DataProviderBase
from .utils import to_one_hot_label

from dotenv import load_dotenv

load_dotenv('./.env')

STRUCTSEG_DIR = os.environ.get('STRUCTSEG_DIR')


class StructSeg2019DataProvider(DataProviderBase):

    Han_Naso_data_format = {
        "channels": 1,
        "depth": 152,
        "height": 512,
        "width": 512,
        "class_num": 23,
    }

    Thoracic_Lung_data_format = {
        "depth": 127,
        "height": 512,
        "width": 512,
        "class_num": 23,
    }
    common_data_format = {
        "channels": 1,
        "height": 512,
        "width": 512,
        "class_num": 23,
    }

    DIR_HUB = {
        'HaN': (
            f"{STRUCTSEG_DIR}/HaN_OAR",
            {
                **common_data_format,
                "depth": 152,
                "class_num": 23,
            },
        ),
        'Naso': (
            f"{STRUCTSEG_DIR}/Naso_GTV",
            {
                **common_data_format,
                "depth": 152,
                "class_num": 2,
            },
        ),
        'Thoracic': (
            f"{STRUCTSEG_DIR}/Thoracic_OAR",
            {
                **common_data_format,
                "depth": 127,
                "class_num": 7,
            },
        ),
        'Lung': (
            f"{STRUCTSEG_DIR}/Lung_GTV",
            {
                **common_data_format,
                "depth": 127,
                "class_num": 2,
            },
        ),
    }

    def __init__(self, args):
        self.data_dir, self._data_format = self.DIR_HUB[args]
        self.all_ids = os.listdir(self.data_dir)
        self.train_ids = self.all_ids[: -len(self.all_ids) // 10]
        self.test_ids = self.all_ids[-len(self.all_ids) // 10:]

    def _get_raw_data_generator(self, data_ids, **kwargs):
        return StructSegGenerator(data_ids, self.data_format, data_dir=self.data_dir, **kwargs)

    @property
    def data_format(self) -> dict:
        return self._data_format


class StructSegGenerator(DataGeneratorBase):

    def __init__(self, data_ids, data_format, data_dir, random=True, preload=False, **kwargs):
        super().__init__(data_ids, data_format, random)
        self.data_dir = data_dir
        self.preload = preload
        if preload:
            self.all_volumes = {}
            self.all_labels = {}
            self.all_affines = {}
            self._preload()

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
        affines = []
        for idx, data_id in enumerate(data_ids):
            if self.preload:
                volume = self.all_volumes[data_id]
                label = self.all_labels[data_id]
                affine = self.all_affines[data_id]
            else:
                volume, label, affine = self._preload_get_image_and_label(data_id)

            label = to_one_hot_label(
                label,
                self.data_format['class_num'],
            )
            batch_volume[idx, :, :len(volume)] = volume[:self.data_format['depth']]
            batch_label[idx, :, :len(volume)] = label[:, :self.data_format['depth']]
            affines.append(affine)

        return {
            'volume': batch_volume,
            'label': batch_label,
            'data_ids': data_ids,
            'affines': affines,
        }

    def _preload(self):
        print('Preloading Data-Generator')
        for data_id in tqdm(self.data_ids):
            self.all_volumes[data_id], self.all_labels[data_id], self.all_affines[data_id] =\
                self._preload_get_image_and_label(data_id)

    def _preload_get_image_and_label(self, data_id):
        # Dims: (N, C, D, H, W)
        img_path = os.path.join(self.data_dir, f"{data_id}/data.nii.gz")
        image_obj = nib.load(img_path)
        affine = image_obj.affine
        image = image_obj.get_fdata()
        image = np.transpose(image, (2, 0, 1))
        label_path = os.path.join(self.data_dir, f"{data_id}/label.nii.gz")

        if os.path.exists(label_path):
            label = nib.load(label_path).get_fdata()
            label = np.transpose(label, (2, 0, 1))
        else:
            label = None
        return image, label, affine
