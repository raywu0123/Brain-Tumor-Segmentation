import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .base import PytorchModelBase
from .loss_functions import ce_minus_log_dice
from .utils import get_tensor_from_array


def activation_fn():
    return nn.ReLU()


class VNet(PytorchModelBase):

    def __init__(
            self,
            data_format: dict,
            duplication_num: int = 8,
            kernel_size: int = 5,
            conv_time: int = 2,
            n_layer: int = 4,
            batch_sampler_id='three_dim',
        ):
        super(VNet, self).__init__(
            batch_sampler_id=batch_sampler_id,
            loss_fn=ce_minus_log_dice,
        )
        # To work properly, kernel_size must be odd
        if kernel_size % 2 == 0:
            raise AssertionError('kernel_size({}) must be odd'.format(kernel_size))
        self.n_layer = n_layer

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        self.duplicate = Duplicate(data_format['channels'], duplication_num, kernel_size)
        for i in range(n_layer):
            n_channel = (2 ** i) * duplication_num
            down_conv = DownConv(n_channel, kernel_size, conv_time)
            self.down.append(down_conv)

        for i in range(n_layer - 1):
            n_channel = (2 ** i) * duplication_num
            up_conv = UpConv(n_channel * 4, n_channel, kernel_size, conv_time)
            self.up.append(up_conv)

        n_channel = (2 ** (n_layer - 1)) * duplication_num
        up_conv = UpConv(n_channel * 2, n_channel, kernel_size, conv_time)
        self.up.append(up_conv)
        self.output_layer = OutLayer(duplication_num * 2, data_format['class_num'])

    def forward(self, x):
        x = get_tensor_from_array(x)
        if x.dim() != 5:
            raise AssertionError('input must have shape (batch_size, channel, D, H, W),\
                                 but get {}'.format(x.shape))

        x_out = []

        x = self.duplicate(x)
        x_out.append(x)

        for down_layer in self.down:
            x = down_layer(x)
            x_out.append(x)

        x_out = x_out[:-1]
        for x_down, u in zip(x_out[::-1], self.up[::-1]):
            x = u(x, x_down)

        x = self.output_layer(x)
        x = F.softmax(x, dim=1)
        return x


###########################################################
#             DnConv                                      #
#  input   [batch_num, input_channel,   D,   H,   W]      #
#  output  [batch_num, output_channel,  D/2, H/2, W/2]    #
###########################################################
class DownConv(nn.Module):

    def __init__(self, input_channel, kernel_size, conv_time):
        super(DownConv, self).__init__()
        output_channel = input_channel * 2
        self.down_conv = nn.Conv3d(input_channel, output_channel, kernel_size=2, stride=2)
        self.activation = activation_fn()
        self.batch_norm = nn.BatchNorm3d(output_channel)
        self.conv_N_time = ConvNTimes(output_channel, kernel_size, conv_time)

    def forward(self, x):
        x = self.down_conv(x)
        x = self.activation(x)
        x1 = self.batch_norm(x)
        x = self.conv_N_time(x1)
        x = x + x1
        return x


###########################################################
#             UpConv                                      #
#  x1      [batch_num, x1_channel,    D/2, H/2, W/2]      #
#  x2      [batch_num, x2_channel,    D,   H,   W]        #
#  output  [batch_num, x2_channel*2,  D*2, H*2, W*2]      #
###########################################################
class UpConv(nn.Module):

    def __init__(self, x1_channel, x2_channel, kernel_size, conv_time):
        super(UpConv, self).__init__()
        self.up_conv = nn.ConvTranspose3d(x1_channel, x2_channel, kernel_size=2, stride=2)
        self.activation = activation_fn()
        self.batch_norm = nn.BatchNorm3d(x2_channel)
        self.conv_N_time = ConvNTimes(x2_channel * 2, kernel_size, conv_time)

    def forward(self, x1, x2):
        x1 = self.up_conv(x1)
        x1 = self.activation(x1)
        x1 = self.batch_norm(x1)
        if x1.shape != x2.shape:
            # this case will only happend for
            # x1 [N, C, D-1, H-1, W-1]
            # x2 [N, C, D,   H,   W  ]
            p_d = x2.shape[2] - x1.shape[2]
            p_h = x2.shape[3] - x1.shape[3]
            p_w = x2.shape[4] - x1.shape[4]
            pad = nn.ConstantPad3d((0, p_w, 0, p_h, 0, p_d), 0)
            x1 = pad(x1)

        x = torch.cat((x1, x2), 1)
        x1 = self.conv_N_time(x)
        x = x1 + x
        return x


###########################################################
#             Conv_N_time                                 #
#  input   [batch_num, channel_num,   D,   H,   W]        #
#  output  [batch_num, channel_num,   D,   H,   W]        #
###########################################################
class ConvNTimes(nn.Module):

    def __init__(self, channel_num, kernel_size, N):
        super(ConvNTimes, self).__init__()

        self.convs = nn.ModuleList()
        self.batchnorms = nn.ModuleList()

        self.activation = activation_fn()
        for _ in range(N):
            conv = nn.Conv3d(channel_num, channel_num, kernel_size=kernel_size,
                             padding=kernel_size // 2)
            self.convs.append(conv)
            norm = nn.BatchNorm3d(channel_num)
            self.batchnorms.append(norm)

    def forward(self, x):
        for conv, batchnorm in zip(self.convs, self.batchnorms):
            x = conv(x)
            x = self.activation(x)
            x = batchnorm(x)
        return x


###########################################################
#             Duplication                                 #
#  input   [batch_num, input_channel,    D,   H,   W]     #
#  output  [batch_num, duplication_num,  D,   H,   W]     #
###########################################################
class Duplicate(nn.Module):

    def __init__(self, input_channel, duplication_num, kernel_size):
        super(Duplicate, self).__init__()
        self.duplicate = nn.Conv3d(input_channel, duplication_num,
                                   kernel_size=kernel_size, padding=kernel_size // 2)
        self.activation = activation_fn()
        self.batch_norm = nn.BatchNorm3d(duplication_num)

    def forward(self, inp):
        x = self.duplicate(inp)
        x = self.activation(x)
        x = self.batch_norm(x)
        return x


###########################################################
#             Out_layer                                   #
#  input   [batch_num, duplication_num*2,  D,   H,   W]   #
#  output  [batch_num, 2,                  D,   H,   W]   #
###########################################################
class OutLayer(nn.Module):

    def __init__(self, input_channel, class_num):
        super(OutLayer, self).__init__()
        self.conv = nn.Conv3d(input_channel, class_num, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
