from .ntu_mri import NTU_MRI
from .brats2015 import BRATS2015

#   DataProviders = {
#       'ntu_mri': (NTU_MRI, NTU_MRI_DIR),
#       'ntu_mock_test': (NTU_MRI, NTU_MOCK_TEST_DIR),
#       'ntu_test': (NTU_MRI, NTU_TEST_DIR),
#
#       'brats2015_hgg': (BRATS2015, [BRATS2015_HGG_DIR]),
#       'brats2015_lgg': (BRATS2015, [BRATS2015_LGG_DIR]),
#       'brats2015': (BRATS2015, [BRATS2015_LGG_DIR, BRATS2015_HGG_DIR])
#   }


def DataProviders(key):
    args = key.split('_')
    data_source = args[0]
    del args[0]
    Providers = {
        'ntu': (NTU_MRI, args),
        'brats2015': (BRATS2015, args)
    }
    return Providers[data_source]
