from functools import partial

from .toy_model import ToyModel
<<<<<<< HEAD
from .u_net import UNet
=======
from .UNet import UNet
>>>>>>> update to date

DEFAULT_TRAINING_PARAM = {
    'batch_size': 50,
    'epoch_num': 1500 * 200,
    'verbose_epoch_num': 10,
}


MODELS = {
    'toy_model': (
        partial(
            ToyModel,
        ),
        DEFAULT_TRAINING_PARAM,
    ),
    'toy_model_big': (
        partial(
            ToyModel,
            **{
                'num_units': (64, 128, 256, 512, 1024),
                'pooling_layer_num': (0, 1, 2),
            },
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
            'batch_size': 25,
        },
    ),
    'UNet': (
        partial(
            UNet,
            **{
                'floor_num': 4,
                'channel_num': 64,
                'conv_times': 2,
            },
        ),
        {
            'batch_size': 25,
            'epoch_num': 60000,
            'verbose_epoch_num': 10,
        },
    ),
}
