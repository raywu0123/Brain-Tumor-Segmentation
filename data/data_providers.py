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


#   DataProviders = {
#       'ntu_mri': (NTU_MRI, NTU_MRI_DIR),
#       'ntu_mock_test': (NTU_MRI, NTU_MOCK_TEST_DIR),
#       'ntu_test': (NTU_MRI, NTU_TEST_DIR),
#
#       'brats2015_hgg': (BRATS2015, [BRATS2015_HGG_DIR]),
#       'brats2015_lgg': (BRATS2015, [BRATS2015_LGG_DIR]),
#       'brats2015': (BRATS2015, [BRATS2015_LGG_DIR, BRATS2015_HGG_DIR])
#   }


def get_ntu_data(key):
    if key.find('mri'):
        return NTU_MRI(NTU_MRI_DIR)
    elif key.find('mock'):
        return NTU_MRI(NTU_MOCK_TEST_DIR)
    elif key.find('test'):
        return NTU_MRI(NTU_TEST_DIR)


def get_modality(args):
    modal_bases = []
    if 'Flair' in args:
        modal_bases.append('Flair.')
    if 'T1' in args:
        modal_bases.append('T1.')
    if 'T1c' in args:
        modal_bases.append('T1c.')
    if 'T2c' in args:
        modal_bases.append('T2.')
    return modal_bases


def get_brats_dir(args):
    data_dirs = []
    if 'hgg' in args:
        data_dirs.append(BRATS2015_HGG_DIR)
    if 'lgg' in args:
        data_dirs.append(BRATS2015_LGG_DIR)
    return data_dirs


def get_brats_data(key):
    modal_bases = ['Flair.', 'T1.', 'T1c.', 'T2.']
    data_dirs = [BRATS2015_LGG_DIR, BRATS2015_HGG_DIR]
    if key == 'brats2015':
        return BRATS2015(modal_bases, data_dirs)
    else:
        args = key.split('_')
        modal_bases = get_modality(args)
        data_dirs = get_brats_dir(args)
        return BRATS2015(modal_bases, data_dirs)


def get_data_provider(key):
    if key.find('ntu'):
        return get_ntu_data(key)
    elif key.find('brats2015'):
        return get_brats_data(key)
