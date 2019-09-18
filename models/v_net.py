import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PytorchModelBase
from .utils import get_tensor_from_array


class VNet(PytorchModelBase):

    def __init__(
            self,
            data_format: dict,
            duplication_num: int = 16,
            kernel_size: int = 3,
            conv_time: int = 2,
            n_layer: int = 4,
            batch_sampler_id='three_dim',
            dropout_rate: float = 0.,
            use_position: bool = False,
            **kwargs,
    ):
        self.use_position = use_position
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        super(VNet, self).__init__(
            batch_sampler_id=batch_sampler_id,
            data_format=data_format,
            head_outcome_channels=duplication_num,
            forward_outcome_channels=duplication_num,
            **kwargs,
        )
        # To work properly, kernel_size must be odd
        if kernel_size % 2 == 0:
            raise AssertionError('kernel_size({}) must be odd'.format(kernel_size))

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()

        for i in range(n_layer):
            n_channel = (2 ** i) * duplication_num
            down_conv = DownConv(n_channel, kernel_size, conv_time, use_position)
            self.down.append(down_conv)

        for i in range(n_layer - 1):
            n_channel = (2 ** i) * duplication_num
            up_conv = UpConv(n_channel * 2, n_channel, kernel_size, conv_time, use_position)
            self.up.append(up_conv)

        n_channel = (2 ** (n_layer - 1)) * duplication_num
        in_channel = n_channel * 2
        up_conv = UpConv(in_channel, n_channel, kernel_size, conv_time, use_position)
        self.up.append(up_conv)

    def forward_head(self, inp, data_idx):
        x, position = inp['volume'], inp['position']
        if x.ndim != 5:
            raise AssertionError('input must have shape (batch_size, channel, D, H, W),\
                                 but get {}'.format(x.shape))
        x = get_tensor_from_array(x)
        if self.use_position:
            pos_vec = get_tensor_from_array(position)
            pos_vec = pos_vec.view(pos_vec.shape[0], pos_vec.shape[1], 1, 1, 1)
            pos_vec = pos_vec.expand(-1, -1, x.shape[-3], x.shape[-2], x.shape[-1])
            x = torch.cat([x, pos_vec], dim=1)

        return self.heads[data_idx](x), position

    def forward(self, inp):
        x, position = inp
        x_out = [x]
        for down_layer in self.down:
            x = down_layer(x, position)
            x_out.append(x)

        x_out = x_out[:-1]
        for x_down, u in zip(x_out[::-1], self.up[::-1]):
            x = u(x, x_down, position)
        return x

    def build_heads(self, input_channels: list, output_channel: int):
        if self.use_position:
            input_channels = [c + 3 for c in input_channels]
        return nn.ModuleList([
            Duplicate(
                input_channel,
                output_channel,
                self.kernel_size,
            )
            for input_channel in input_channels
        ])

    def build_tails(self, input_channels, class_nums):
        return nn.ModuleList([
            nn.Conv3d(input_channels, class_num, kernel_size=1)
            for class_num in class_nums
        ])


###########################################################
#             DnConv                                      #
#  input   [batch_num, input_channel,   D,   H,   W]      #
#  output  [batch_num, output_channel,  D/2, H/2, W/2]    #
###########################################################
class DownConv(nn.Module):

    def __init__(self, input_channel, kernel_size, conv_time, use_position):
        super(DownConv, self).__init__()
        output_channel = input_channel * 2
        self.use_position = use_position
        if self.use_position:
            input_channel += 3
        self.down_conv = nn.Conv3d(input_channel, output_channel, kernel_size=kernel_size, stride=2)
        self.conv_N_time = ConvNTimes(output_channel, kernel_size, conv_time)

    def forward(self, x, position):
        if self.use_position:
            pos_vec = get_tensor_from_array(position)
            pos_vec = pos_vec.view(pos_vec.shape[0], pos_vec.shape[1], 1, 1, 1)
            pos_vec = pos_vec.expand(-1, -1, x.shape[-3], x.shape[-2], x.shape[-1])
            x = torch.cat([x, pos_vec], dim=1)

        x = self.down_conv(x)
        x = F.relu(x)
        x = self.conv_N_time(x)
        return x


###########################################################
#             UpConv                                      #
#  x1      [batch_num, x1_channel,    D/2, H/2, W/2]      #
#  x2      [batch_num, x2_channel,    D,   H,   W]        #
#  output  [batch_num, x2_channel*2,  D*2, H*2, W*2]      #
###########################################################
class UpConv(nn.Module):

    def __init__(self, x1_channel, x2_channel, kernel_size, conv_time, use_position):
        super(UpConv, self).__init__()
        self.use_position = use_position
        if self.use_position:
            x1_channel += 3
        self.up_conv = nn.ConvTranspose3d(x1_channel, x2_channel, kernel_size=kernel_size, stride=2)
        self.conv_N_time = ConvNTimes(x2_channel, kernel_size, conv_time)

    def forward(self, x, x_down, position):
        if self.use_position:
            pos_vec = get_tensor_from_array(position)
            pos_vec = pos_vec.view(pos_vec.shape[0], pos_vec.shape[1], 1, 1, 1)
            pos_vec = pos_vec.expand(-1, -1, x.shape[-3], x.shape[-2], x.shape[-1])
            x = torch.cat([x, pos_vec], dim=1)
        x = self.up_conv(x)
        x = F.relu(x)
        if x.shape != x_down.shape:
            # this case will only happen when
            # x1 [N, C, D-1, H-1, W-1]
            # x2 [N, C, D,   H,   W  ]
            p_d = x_down.shape[2] - x.shape[2]
            p_h = x_down.shape[3] - x.shape[3]
            p_w = x_down.shape[4] - x.shape[4]
            pad = nn.ConstantPad3d((0, p_w, 0, p_h, 0, p_d), 0)
            x = pad(x)

        x = x + x_down
        x = self.conv_N_time(x)
        return x


###########################################################
#             Conv_N_time                                 #
#  input   [batch_num, channel_num,   D,   H,   W]        #
#  output  [batch_num, channel_num,   D,   H,   W]        #
###########################################################
class ConvNTimes(nn.Module):

    def __init__(self, channel_num, kernel_size, N):
        super(ConvNTimes, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv3d(
                channel_num,
                channel_num,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            for _ in range(N)
        ])
        self.norms = nn.ModuleList([
            nn.InstanceNorm3d(num_features=channel_num)
            for _ in range(N)
        ])

    def forward(self, inp):
        x = inp
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = norm(x)
            x = F.relu(x)
        x += inp
        return x


###########################################################
#             Duplication                                 #
#  input   [batch_num, input_channel,    D,   H,   W]     #
#  output  [batch_num, duplication_num,  D,   H,   W]     #
###########################################################
class Duplicate(nn.Module):

    def __init__(self, input_channel, duplication_num, kernel_size):
        super(Duplicate, self).__init__()
        self.duplicate = nn.Conv3d(
            input_channel,
            duplication_num,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

    def forward(self, inp):
        x = self.duplicate(inp)
        x = F.relu(x)
        return x
