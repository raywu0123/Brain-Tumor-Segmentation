import os

from dotenv import load_dotenv

from .ntu_mri import NTU_MRI

load_dotenv('./.env')

NTU_MRI_DIR = os.environ.get('NTU_MRI_DIR')
NTU_MOCK_TEST_DIR = os.environ.get('NTU_MOCK_TEST_DIR')
NTU_TEST_DIR = os.environ.get('NTU_TEST_DIR')

DataProviders = {
    'ntu_mri': (NTU_MRI, NTU_MRI_DIR),
    'ntu_mock_test': (NTU_MRI, NTU_MOCK_TEST_DIR),
    'ntu_test': (NTU_MRI, NTU_TEST_DIR),
}
