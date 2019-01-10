from functools import partial

from .toy_model import ToyModel
from .u_net import UNet
from .v_net import VNet

DEFAULT_TRAINING_PARAM = {
    'batch_size': 50,
    'epoch_num': 1500 * 200,
    'verbose_epoch_num': 10,
}


ModelHub = {
    'toy_model': (
        partial(
            ToyModel,
        ),
        DEFAULT_TRAINING_PARAM,
    ),
    'toy_model_big': (
        partial(
            ToyModel,
            num_units=(64, 128, 256, 512, 1024),
            pooling_layer_num=(0, 1, 2),
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 25,
        },
    ),
    'u_net': (
        partial(
            UNet,
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 20,
        },
    ),
    'v_net': (
        partial(
            VNet,
            duplication_num=8,
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 1,
        },
    ),
    'v_net_patch': (
        partial(
            VNet,
            batch_sampler_id='center_patch3d'
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 20,
        },
    ),
}
