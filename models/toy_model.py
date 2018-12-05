import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dotenv import load_dotenv

from .base import Model2DBase

load_dotenv('./.env')
RESULT_DIR_BASE = os.environ.get('RESULT_DIR')


class ToyModel(Model2DBase):

    def __init__(
            self,
            data_format: dict,
            num_units: [int] = (32, 32, 64, 64, 128),
            pooling_layer_num: [int] = (1, 3),
            kernel_size: int = 3,
            lr: float = 1e-4,
        ):
        super(ToyModel, self).__init__(data_format)
        self.model = ToyModelNet(
            data_format=data_format,
            num_units=num_units,
            pooling_layer_num=pooling_layer_num,
            kernel_size=kernel_size,
        )
        self.opt = optim.Adam(params=self.model.parameters(), lr=lr)
        if torch.cuda.is_available():
            self.model.cuda()


class ToyModelNet(nn.Module):

    def __init__(
        self,
        data_format,
        num_units,
        kernel_size,
        pooling_layer_num
    ):
        super(ToyModelNet, self).__init__()
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
        x = inp
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
