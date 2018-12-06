import os

from dotenv import load_dotenv

from .ntu_mri import NtuMriDataProvider
from .brats2015 import Brats2015DataProvider

load_dotenv('./.env')

NTU_MRI_DIR = os.environ.get('NTU_MRI_DIR')
NTU_MOCK_TEST_DIR = os.environ.get('NTU_MOCK_TEST_DIR')
NTU_TEST_DIR = os.environ.get('NTU_TEST_DIR')

BRATS2015_DIR = os.environ.get('BRATS2015_DIR')
BRATS2015_HGG_DIR = os.path.join(BRATS2015_DIR, './HGG')
BRATS2015_LGG_DIR = os.path.join(BRATS2015_DIR, './LGG')

DataProviderHub = {
    'ntu_mri': (NtuMriDataProvider, NTU_MRI_DIR),
    'ntu_mock_test': (NtuMriDataProvider, NTU_MOCK_TEST_DIR),
    'ntu_test': (NtuMriDataProvider, NTU_TEST_DIR),

    'brats2015_hgg': (Brats2015DataProvider, [BRATS2015_HGG_DIR]),
    'brats2015_lgg': (Brats2015DataProvider, [BRATS2015_LGG_DIR]),
    'brats2015': (Brats2015DataProvider, [BRATS2015_LGG_DIR, BRATS2015_HGG_DIR])
}
