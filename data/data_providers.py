from .ntu_mri import NtuMriDataProvider
from .brats2015 import Brats2015DataProvider


class DataProviderHub:

    def __init__(self):
        self.ProviderHub = {
            'ntu': NtuMriDataProvider,
            'brats2015': Brats2015DataProvider
        }

    def __getitem__(self, key):
        key = key.split('_')
        data_source, args = key[0], key[1:]
        args = '_'.join(args)
        return self.ProviderHub[data_source], args
