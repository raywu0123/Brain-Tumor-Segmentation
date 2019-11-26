from functools import partial

from .toy_model import ToyModel
from .u_net import UNet
from .v_net import VNet
from .pspnet import PSPNet
from .high_resolution_compact_network import HighResolutionCompactNetwork

DEFAULT_TRAINING_PARAM = {
    'batch_size': 50,
    'epoch_num': 1500 * 200,
    'verbose_epoch_num': 0.2,
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
    'u_net_positional': (
        partial(
            UNet,
            use_position=True,
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 20,
        },
    ),
    'u_net_positional_HaN': (
        partial(
            UNet,
            floor_num=5,
            conv_times=3,
            use_position=True,
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 2,
        },
    ),
    'u_net_positional_dropout0.1': (
        partial(
            UNet,
            use_position=True,
            dropout_rate=0.1,
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 2,
        },
    ),
    'u_net_positional_HaN_attention': (
        partial(
            UNet,
            floor_num=5,
            conv_times=3,
            attention=True,
            use_position=True,
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 2,
        },
    ),
    'u_net_positional_HaN_self_attention': (
        partial(
            UNet,
            floor_num=5,
            conv_times=3,
            self_attention=3,
            use_position=True,
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 2,
        },
    ),
    'v_net': (
        partial(
            VNet,
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 1,
        },
    ),
    'v_net_dropout0.1': (
        partial(
            VNet,
            dropout_rate=0.1,
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 1,
        },
    ),
    'v_net_uniform_patch': (
        partial(
            VNet,
            batch_sampler_id='uniform_patch3d',
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 15,
        },
    ),
    'v_net_center_patch': (
        partial(
            VNet,
            batch_sampler_id='center_patch_96',
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 15,
        },
    ),
    'pspnet_2d_resnet34': (
        partial(
            PSPNet,
            backend='resnet34'
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 40,
        },
    ),
    'pspnet_2d_resnet50': (
        partial(
            PSPNet,
            backend='resnet50',
            psp_size=2048,
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 20,
        },
    ),
    'pspnet_2d_resnet101': (
        partial(
            PSPNet,
            backend='resnet101',
            psp_size=2048,
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 10,
        },
    ),
    'pspnet_2d_resnet152': (
        partial(
            PSPNet,
            backend='resnet152',
            psp_size=2048,
        ),
        {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 10,
        },
    ),
    'HighRes3DNet': (
        partial(
            HighResolutionCompactNetwork,
        ), {
            **DEFAULT_TRAINING_PARAM,
            'batch_size': 4,
        }
    )
}
