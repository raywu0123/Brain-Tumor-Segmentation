from unittest import TestCase
from ..data_providers import DataProviderHub


class DataGeneratorTestCase(TestCase):

    def setUp(self):
        data_provider_hub = DataProviderHub()
        get_ntu_provider, args = data_provider_hub['ntu_mri']
        self.ntu_provider = get_ntu_provider(args)
