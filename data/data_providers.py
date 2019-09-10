from .ntu_mri import NtuMriDataProvider
from .brats2015 import Brats2015DataProvider
from .structseg import StructSeg2019DataProvider
from .tcia_ct import TCIACTDataProvider


class DataProviderHub:

    def __init__(self):
        self.ProviderHub = {
            'ntu': NtuMriDataProvider,
            'brats2015': Brats2015DataProvider,
            'struct': StructSeg2019DataProvider,
            'tciact': TCIACTDataProvider,
        }

    def __getitem__(self, key):
        key = key.split('_')
        data_source, args = key[0], key[1:]
        args = '_'.join(args)
        return self.ProviderHub[data_source], args
