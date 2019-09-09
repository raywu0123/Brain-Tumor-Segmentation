import os

from tqdm import tqdm
import nrrd
import numpy as np

np.random.seed = 0

from .base import DataGeneratorBase
from .data_provider_base import DataProviderBase

from dotenv import load_dotenv

load_dotenv('./.env')


TCIA_CT_DIR = os.environ.get('TCIA_CT_DIR')
TEST_DIR = os.path.join(TCIA_CT_DIR, 'nrrds', 'test')
VAL_DIR = os.path.join(TCIA_CT_DIR, 'nrrds', 'validation')


class TCIACTDataProvider(DataProviderBase):

    _data_format = {
        "channels": 1,
        "depth": 180,
        "height": 512,
        "width": 512,
        "class_num": 22,
    }

    def __init__(self, mode='', **kwargs):
        """
        :param mode: ['oncologist', 'radiographer', '']
        empty string for mode means both
        """
        # TODO
        self.mode = mode

        self.train_ids = os.listdir(os.path.join(TEST_DIR, mode))
        self.train_ids = [os.path.join(TEST_DIR, mode, idx) for idx in self.train_ids]
        self.test_ids = os.listdir(os.path.join(VAL_DIR, mode))
        self.test_ids = [os.path.join(VAL_DIR, mode, idx) for idx in self.test_ids]

    def _get_raw_data_generator(self, data_ids, **kwargs):
        return TCIACTGenerator(data_ids, self.data_format, self.mode, **kwargs)

    @property
    def data_format(self) -> dict:
        return self._data_format


class TCIACTGenerator(DataGeneratorBase):

    def __init__(self, data_ids, data_format, mode, random=True, preload=False, **kwargs):
        super().__init__(data_ids, data_format, random)
        self.mode = mode
        self.preload = preload
        if preload:
            self.all_volumes = {}
            self.all_labels = {}
            self.all_headers = {}
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
        batch_label[:, 0] = 1

        headers = []
        for idx, data_id in enumerate(data_ids):
            if self.preload:
                volume = self.all_volumes[data_id]
                label = self.all_labels[data_id]
                header = self.all_headers[data_id]
            else:
                volume, label, header = self._preload_get_image_and_label(data_id)

            batch_volume[idx, :, :len(volume)] = volume[:self.data_format['depth']]
            batch_label[idx, :len(volume)] = label[:self.data_format['depth']]
            headers.append(header)

        return {
            'volume': batch_volume,
            'label': batch_label,
            'data_ids': data_ids,
            'headers': headers,
        }

    def _preload(self):
        print('Preloading Data-Generator')
        for data_id in tqdm(self.data_ids):
            self.all_volumes[data_id], self.all_labels[data_id], self.all_headers[data_id] =\
                self._preload_get_image_and_label(data_id)

    @staticmethod
    def _preload_get_image_and_label(data_id):
        # Dims: (N, C, D, H, W)
        img_path = os.path.join(data_id, 'CT_IMAGE.nrrd')
        data, header = nrrd.read(img_path)

        segmentations = sorted(os.listdir(os.path.join(data_id, 'segmentations')))
        label = np.zeros_like(data, dtype=np.uint8)
        for class_idx, seg in enumerate(segmentations):
            one_class_label_path = os.path.join(data_id, 'segmentations', seg)
            one_class_label, _ = nrrd.read(one_class_label_path)
            label += one_class_label * (class_idx + 1) * (label == 0)
            # TODO, handle overlapping labels

        data = data.transpose([2, 0, 1])
        label = label.transpose([2, 0, 1])
        return data, label, header
