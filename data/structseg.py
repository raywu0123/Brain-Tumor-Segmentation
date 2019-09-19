import os

from tqdm import tqdm
import nibabel as nib
import numpy as np
np.random.seed = 0

from .base import DataGeneratorBase
from .data_provider_base import DataProviderBase
from metrics import StructSegHaNMetric, NTUMetric

from dotenv import load_dotenv

load_dotenv('./.env')

STRUCTSEG_DIR = os.environ.get('STRUCTSEG_DIR')
STRUCTSEG_TEST_DIR = os.environ.get('STRUCTSEG_TEST_DIR')


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
            StructSegHaNMetric,
        ),
        'Naso': (
            f"{STRUCTSEG_DIR}/Naso_GTV",
            {
                **common_data_format,
                "depth": 152,
                "class_num": 2,
            },
            NTUMetric,
        ),
        'Thoracic': (
            f"{STRUCTSEG_DIR}/Thoracic_OAR",
            {
                **common_data_format,
                "depth": 127,
                "class_num": 7,
            },
            NTUMetric,
        ),
        'Lung': (
            f"{STRUCTSEG_DIR}/Lung_GTV",
            {
                **common_data_format,
                "depth": 127,
                "class_num": 2,
            },
            NTUMetric,
        ),
    }

    def __init__(self, args: str):
        is_test = False
        if args.endswith('_test'):
            args = args[:-5]
            is_test = True

        self.data_dir, self._data_format, self._metric = self.DIR_HUB[args]
        if is_test:
            self.data_dir = STRUCTSEG_TEST_DIR
        self.all_ids = os.listdir(self.data_dir)
        self.train_ids = self.all_ids[: -len(self.all_ids) // 10]
        self.test_ids = self.all_ids[-len(self.all_ids) // 10:]

    def _get_raw_data_generator(self, data_ids, **kwargs):
        return StructSegDataGenerator(data_ids, self.data_format, data_dir=self.data_dir, **kwargs)

    @property
    def data_format(self) -> dict:
        return self._data_format


class StructSegDataGenerator(DataGeneratorBase):

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
        original_shapes = []
        no_label = False
        for idx, data_id in enumerate(data_ids):
            if self.preload:
                volume = self.all_volumes[data_id]
                label = self.all_labels[data_id]
                affine = self.all_affines[data_id]
            else:
                volume, label, affine = self._preload_get_image_and_label(data_id)

            batch_volume[
                idx, :,
                :volume.shape[-3],
                :volume.shape[-2],
                :volume.shape[-1],
            ] = volume[
                :self.data_format['depth'],
                :self.data_format['height'],
                :self.data_format['width'],
            ]
            if label is not None:
                batch_label[
                    idx,
                    :label.shape[-3],
                    :label.shape[-2],
                    :label.shape[-1],
                ] = label[
                    :self.data_format['depth'],
                    :self.data_format['height'],
                    :self.data_format['width'],
                ]
            else:
                no_label = True
            affines.append(affine)
            original_shapes.append(volume.shape)

        data_ids = [f'{data_id}.nii.gz' for data_id in data_ids]
        return {
            'volume': batch_volume,
            'label': batch_label,
            'data_ids': data_ids,
            'affines': affines,
            'original_shapes': original_shapes,
            'no_label': no_label,
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
