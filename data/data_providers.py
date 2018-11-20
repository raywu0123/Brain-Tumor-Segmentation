import os

from dotenv import load_dotenv

from .ntu_mri import NTU_MRI
from .brats2015 import BRATS2015

load_dotenv('./.env')

NTU_MRI_DIR = os.environ.get('NTU_MRI_DIR')
NTU_MOCK_TEST_DIR = os.environ.get('NTU_MOCK_TEST_DIR')
NTU_TEST_DIR = os.environ.get('NTU_TEST_DIR')

BRATS2015_DIR = os.environ.get('BRATS2015_DIR')
BRATS2015_HGG_DIR = os.path.join(BRATS2015_DIR, './HGG')
BRATS2015_LGG_DIR = os.path.join(BRATS2015_DIR, './LGG')

DataProviders = {
    'ntu_mri': (NTU_MRI, NTU_MRI_DIR),
    'ntu_mock_test': (NTU_MRI, NTU_MOCK_TEST_DIR),
    'ntu_test': (NTU_MRI, NTU_TEST_DIR),

    'brats2015_hgg': (BRATS2015, [BRATS2015_HGG_DIR]),
    'brats2015_lgg': (BRATS2015, [BRATS2015_LGG_DIR]),
    'brats2015': (BRATS2015, [BRATS2015_LGG_DIR, BRATS2015_HGG_DIR])
}
