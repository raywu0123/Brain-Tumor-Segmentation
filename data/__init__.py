import os

from dotenv import load_dotenv

from .ntu_mri import NtuMriDataGeneratorFactory
from .brats2015 import Brats2015DataGeneratorFactory

load_dotenv('./.env')

NTU_MRI_DIR = os.environ.get('NTU_MRI_DIR')
NTU_MOCK_TEST_DIR = os.environ.get('NTU_MOCK_TEST_DIR')
NTU_TEST_DIR = os.environ.get('NTU_TEST_DIR')

BRATS2015_DIR = os.environ.get('BRATS2015_DIR')
BRATS2015_HGG_DIR = os.path.join(BRATS2015_DIR, './HGG')
BRATS2015_LGG_DIR = os.path.join(BRATS2015_DIR, './LGG')

DataProviders = {
    'ntu_mri': (NtuMriDataGeneratorFactory, NTU_MRI_DIR),
    'ntu_mock_test': (NtuMriDataGeneratorFactory, NTU_MOCK_TEST_DIR),
    'ntu_test': (NtuMriDataGeneratorFactory, NTU_TEST_DIR),

    'brats2015_hgg': (Brats2015DataGeneratorFactory, [BRATS2015_HGG_DIR]),
    'brats2015_lgg': (Brats2015DataGeneratorFactory, [BRATS2015_LGG_DIR]),
    'brats2015': (Brats2015DataGeneratorFactory, [BRATS2015_LGG_DIR, BRATS2015_HGG_DIR])
}
