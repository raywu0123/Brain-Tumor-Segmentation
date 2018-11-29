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

"""DataProviders = {
    'ntu_mri': (NTU_MRI, NTU_MRI_DIR),
    'ntu_mock_test': (NTU_MRI, NTU_MOCK_TEST_DIR),
    'ntu_test': (NTU_MRI, NTU_TEST_DIR),

    'brats2015_hgg': (BRATS2015, [BRATS2015_HGG_DIR]),
    'brats2015_lgg': (BRATS2015, [BRATS2015_LGG_DIR]),
    'brats2015': (BRATS2015, [BRATS2015_LGG_DIR, BRATS2015_HGG_DIR])
}"""

def get_data_provider(key):
    if key.find('ntu'):
        if key.find('mri'):
            return NTU_MRI(NTU_MRI_DIR)
        elif key.find('mock'):
            return NTU_MRI(NTU_MOCK_TEST_DIR)
        elif key.find('test'):
            return NTU_MRI(NTU_TEST_DIR)
    elif key.find('brats2015'):
        modal_bases = ['Flair.', 'T1.', 'T1c.', 'T2.']
        data_dirs = [BRATS2015_LGG_DIR, BRATS2015_HGG_DIR]
        if key == 'brats2015':
            return BRATS2015(modal_bases, data_dirs)
        if key == 'brats2015_hgg':
            data_dirs = [BRATS2015_HGG_DIR]
            return BRATS2015(modal_bases, data_dirs)
        if key == 'brats2015_lgg':
            data_dirs = [BRATS2015_LGG_DIR]
            return BRATS2015(modal_bases, data_dirs)
        modal_bases = []
        if key.find('Flair'):
            modal_bases.append('Flair.')
        if key.find('T1'):
            modal_bases.append('T1.')
        if key.find('T1c'):
            modal_bases.append('T1c.')
        if key.find('T2c'):
            modal_bases.append('T2.')
        if key.find('hgg'):
            data_dirs = [BRATS2015_HGG_DIR]
            return BRATS2015(modal_bases, data_dirs)
        elif key.find('lgg'):
            data_dirs = [BRATS2015_LGG_DIR]
            return BRATS2015(modal_bases, data_dirs)
        else:
            data_dirs = [BRATS2015_LGG_DIR, BRATS2015_HGG_DIR]
            return BRATS2015(modal_bases, data_dirs)

