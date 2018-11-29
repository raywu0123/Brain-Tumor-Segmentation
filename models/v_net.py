import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from dotenv import load_dotenv

from .utils import weighted_cross_entropy, soft_dice_score, get_tensor_from_array
from .base import ModelBase


load_dotenv('./.env')
RESULT_DIR_BASE = os.environ.get('RESULT_DIR')


def Activation():
    return nn.ReLU()


class Model3DBase(ModelBase):
    def __init__(
            self,
            channels: int = 1,
            depth: int = 200,
            height: int = 200,
            width: int = 200,
            metadata_dim: int = 0,
            class_num: int = 2,
        ):
        super(Model3DBase, self).__init__(
            channels=channels,
            depth=depth,
            height=height,
            width=width,
            metadata_dim=metadata_dim,
            class_num=class_num,
        )

    def train_on_batch(self, training_datagenerator, batch_size):

        batch_data = training_datagenerator(batch_size=batch_size)
        data, label = batch_data['volume'], batch_data['label']
        data = get_tensor_from_array(data)
        class_weight = np.divide(
            1., np.mean(label, axis=(0, 2, 3, 4)),
            out=np.ones(label.shape[1]),
            where=np.mean(label, axis=(0, 2, 3, 4)) != 0,
        )
        label = get_tensor_from_array(label)

        self.model.train()
        pre = self.model(data)
        crossentropy_loss = weighted_cross_entropy(pre, label, class_weight)
        dice_score = soft_dice_score(pre, label)
        total_loss = crossentropy_loss - torch.log(dice_score)

        self.opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.opt.step()

        return crossentropy_loss.cpu().data.numpy(), dice_score.cpu().data.numpy()

    def predict(self, test_data, batch_size, **kwargs):
        self.model.eval()
        data = test_data['volume']
        data = get_tensor_from_array(data)
        pre = self.model(data)
        return pre.cpu().data.numpy()


class VNet(Model3DBase):
    def __init__(
            self,
            channels: int = 1,
            depth: int = 200,
            height: int = 200,
            width: int = 200,
            metadata_dim: int = 0,
            class_num: int = 2,
            lr: float = 1e-4,
            duplication_num: int = 8,
            kernel_size: int = 5,
            conv_time: int = 2,
            n_layer: int = 4,
        ):
        super(VNet, self).__init__(
            channels=channels,
            depth=depth,
            height=height,
            width=width,
            metadata_dim=metadata_dim,
            class_num=class_num,
        )

        self.model = Vnet_net(channels, duplication_num, kernel_size,
                              conv_time, n_layer, class_num)
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self.opt = optim.Adam(params=self.model.parameters(), lr=lr)


###########################################################
#             Vnet_net                                    #
#  input   [batch_num, input_channel,  D,   H,   W]       #
#  output  [batch_num,             2,  D,   H,   W]       #
###########################################################
class Vnet_net(nn.Module):
    def __init__(
            self,
            channels: int = 1,
            duplication_num: int = 8,
            kernel_size: int = 5,
            conv_time: int = 2,
            n_layer: int = 4,
            class_num: int = 2,
        ):
        super(Vnet_net, self).__init__()
        # To work properly, kernel_size must be odd
        if kernel_size % 2 == 0:
            raise AssertionError('kernel_size({}) must be odd'.format(kernel_size))
        self.n_layer = n_layer

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        self.duplicate = Duplicate(channels, duplication_num, kernel_size)
        for i in range(n_layer):
            n_channel = np.power(2, i) * duplication_num
            dnConv = DnConv(n_channel, kernel_size, conv_time)
            self.down.append(dnConv)
        for i in range(n_layer - 1):
            n_channel = np.power(2, i) * duplication_num
            upConv = UpConv(n_channel * 4, n_channel, kernel_size, conv_time)
            self.up.append(upConv)
        n_channel = np.power(2, n_layer - 1) * duplication_num
        upConv = UpConv(n_channel * 2, n_channel, kernel_size, conv_time)
        self.up.append(upConv)
        self.output_layer = Out_layer(duplication_num * 2, class_num)

    def forward(self, inp):
        if inp.dim() != 5:
            raise AssertionError('input must have shape (batch_size, channel, D, H, W),\
                                 but get {}'.format(inp.shape))

        x_out = []

        x = self.duplicate(inp)
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
class DnConv(nn.Module):
    def __init__(self, input_channel, kernel_size, conv_time):
        super(DnConv, self).__init__()
        output_channel = input_channel * 2
        self.dnconv = nn.Conv3d(input_channel, output_channel, kernel_size=2, stride=2)
        self.activation = Activation()
        self.batch_norm = nn.BatchNorm3d(output_channel)
        self.conv_N_time = Conv_N_time(output_channel, kernel_size, conv_time)

    def forward(self, x):
        x = self.dnconv(x)
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
        self.upconv = nn.ConvTranspose3d(x1_channel, x2_channel, kernel_size=2, stride=2)
        self.activation = Activation()
        self.batch_norm = nn.BatchNorm3d(x2_channel)
        self.conv_N_time = Conv_N_time(x2_channel * 2, kernel_size, conv_time)
        self.padding = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x1 = self.activation(x1)
        x1 = self.batch_norm(x1)
        if x1.shape != x2.shape:
            # this case will only happend for
            # x1 [N, C, D-1, H-1, W-1]
            # x2 [N, C, D,   H,   W  ]
            x1 = self.padding(x1)
        x = torch.cat((x1, x2), 1)
        x1 = self.conv_N_time(x)
        x = x1 + x
        return x


###########################################################
#             Conv_N_time                                 #
#  input   [batch_num, channel_num,   D,   H,   W]        #
#  output  [batch_num, channel_num,   D,   H,   W]        #
###########################################################
class Conv_N_time(nn.Module):
    def __init__(self, channel_num, kernel_size, N):
        super(Conv_N_time, self).__init__()

        self.convs = nn.ModuleList()
        self.batchnorms = nn.ModuleList()

        self.activation = Activation()
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
        self.activation = Activation()
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
class Out_layer(nn.Module):
    def __init__(self, input_channel, class_num):
        super(Out_layer, self).__init__()
        self.conv = nn.Conv3d(input_channel, class_num, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x
