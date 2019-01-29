import torch.nn as nn
import torch.nn.functional as F

from .base import PytorchModelBase
from .loss_functions import ce_minus_log_dice
from .utils import get_tensor_from_array, normalize_batch_image


class ToyModel(PytorchModelBase):

    def __init__(
        self,
        data_format: dict,
        num_units: [int] = (32, 32, 64, 64, 128),
        pooling_layer_num: [int] = (1, 3),
        kernel_size: int = 3,
    ):
        super(ToyModel, self).__init__(
            batch_sampler_id='two_dim',
            loss_fn=ce_minus_log_dice,
            data_format=data_format,
        )
        self.image_chns = data_format['channels']
        self.image_height = data_format['height']
        self.image_width = data_format['width']
        self.class_num = data_format['class_num']

        encoder_num_units = (self.image_chns,) + num_units
        self.encoder_convs = nn.ModuleList()
        self.encoder_batchnorms = nn.ModuleList()
        for idx in range(len(encoder_num_units) - 1):
            if idx in pooling_layer_num:
                stride = 2
            else:
                stride = 1
            in_chns = encoder_num_units[idx]
            out_chns = encoder_num_units[idx + 1]
            conv = nn.Conv2d(
                in_chns,
                out_chns,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
            batchnorm = nn.BatchNorm2d(in_chns)
            self.encoder_convs.append(conv)
            self.encoder_batchnorms.append(batchnorm)

        decoder_num_units = num_units[::-1] + (self.class_num,)
        self.decoder_deconvs = nn.ModuleList()
        self.decoder_batchnorms = nn.ModuleList()
        img_size = self.image_height // (2 ** len(pooling_layer_num))
        for idx in range((len(decoder_num_units)) - 1):
            if idx in pooling_layer_num:
                stride = 2
                img_size = img_size * 2
                output_padding = 1
            else:
                stride = 1
                output_padding = 0
            deconv = nn.ConvTranspose2d(
                decoder_num_units[idx],
                decoder_num_units[idx + 1],
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                output_padding=output_padding,
            )
            batchnorm = nn.BatchNorm2d(decoder_num_units[idx])
            self.decoder_deconvs.append(deconv)
            self.decoder_batchnorms.append(batchnorm)

    def forward(self, inp):
        x = normalize_batch_image(inp)
        x = get_tensor_from_array(x)
        for conv, batchnorm in zip(self.encoder_convs, self.encoder_batchnorms):
            x = batchnorm(x)
            x = F.relu(x)
            x = conv(x)

        for deconv, batchnorm in zip(self.decoder_deconvs, self.decoder_batchnorms):
            x = batchnorm(x)
            x = F.relu(x)
            x = deconv(x)

        x = F.softmax(x, dim=1)
        return x
