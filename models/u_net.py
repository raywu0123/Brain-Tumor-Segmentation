import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PytorchModelBase
from .utils import get_tensor_from_array, normalize_batch_image


class UNet(PytorchModelBase):

    def __init__(
        self,
        data_format: dict,
        batch_sampler_id: str = 'two_dim',
        floor_num: int = 4,
        kernel_size: int = 3,
        channel_num: int = 64,
        conv_times: int = 2,
        **kwargs,
    ):
        super(UNet, self).__init__(
            batch_sampler_id=batch_sampler_id,
            data_format=data_format,
            **kwargs,
        )
        self.floor_num = floor_num
        image_chns, class_num = data_format['channels'], data_format['class_num']
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        in_conv = ConvNTimes(image_chns, channel_num, kernel_size, conv_times)
        self.down_layers.append(in_conv)
        for floor_idx in range(floor_num):
            channel_times = 2 ** floor_idx
            d = DownConv(channel_num * channel_times, kernel_size, conv_times)
            self.down_layers.append(d)

        for floor_idx in range(floor_num)[::-1]:
            channel_times = 2 ** floor_idx
            u = UpConv(channel_num * 2 * channel_times, kernel_size, conv_times)
            self.up_layers.append(u)
        self.out_conv = nn.Conv2d(channel_num, class_num, kernel_size=1)

    def forward(self, inp):
        x = normalize_batch_image(inp)
        x = get_tensor_from_array(x)

        x_out = []
        for down_layer in self.down_layers:
            x = down_layer(x)
            x_out.append(x)
        x_out = x_out[:-1]
        for x_down, u in zip(x_out[::-1], self.up_layers):
            x = u(x, x_down)
        x = self.out_conv(x)
        x = F.softmax(x, dim=1)
        return x


class ConvNTimes(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, conv_times):
        super(ConvNTimes, self).__init__()
        assert(conv_times > 0)
        self.convs = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.conv_times = conv_times

        conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.convs.append(conv)

        for _ in range(conv_times - 1):
            conv = nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
            self.convs.append(conv)
        for _ in range(conv_times):
            batchnorm = nn.BatchNorm2d(out_ch)
            self.batchnorms.append(batchnorm)

    def forward(self, x):
        for conv, batchnorm in zip(self.convs, self.batchnorms):
            x = conv(x)
            x = F.relu(x)
            x = batchnorm(x)
        return x


class DownConv(nn.Module):

    def __init__(self, in_ch, kernel_size, conv_times):
        out_ch = in_ch * 2
        super(DownConv, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvNTimes(in_ch, out_ch, kernel_size, conv_times)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_ch, kernel_size, conv_times):
        super(UpConv, self).__init__()
        out_ch = in_ch // 2
        self.conv_transpose = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size,
            padding=kernel_size // 2,
            stride=2,
        )
        self.conv = ConvNTimes(in_ch, out_ch, kernel_size, conv_times)

    def forward(self, x_down, x_up):
        x_down = self.conv_transpose(x_down)
        diff_x = x_up.size()[2] - x_down.size()[2]
        diff_y = x_up.size()[3] - x_down.size()[3]
        x_down = F.pad(x_down, (0, diff_x, 0, diff_y))
        x = torch.cat([x_down, x_up], dim=1)
        x = self.conv(x)
        return x
