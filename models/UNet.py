import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Model2DBase
from dotenv import load_dotenv
load_dotenv('./.env')


class UNet(Model2DBase):
    def __init__(
        self,
        channels: int = 1,
        kernel_size: int = 3,
        channel_num: int = 48,
        floor_num: int = 4,
    ):
        super(UNet, self).__init__()
        self.floor_num = floor_num
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        self.inc = inconv(channels, channel_num, kernel_size)

        for i in range(floor_num):
            d = down(channel_num, channel_num * 2, kernel_size)
            channel_num = int(channel_num * 2)
            self.down_layers.append(d)

        channel_num = channel_num * 2
        for i in range(floor_num):
            u = up(channel_num, int(channel_num / 2), kernel_size)
            channel_num = int(channel_num / 2)
            self.up_layers.append(u)

        self.outc = outconv(channel_num)

    def forward(self, x):
        x_out = []

        x = self.inc(x)
        x_out.append(x)

        for i in range(self.floor_num):
            x = self.down_layers[i](x)
            x_out.append(x)

        for i in range(self.floor_num):
            x = self.up_layers[i](x, x_out[self.floor_num - 1 - i])

        x = self.outc(x)

        x = torch.sigmoid(x)
        return x


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, kernel_size)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch, kernel_size)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, bilinear=True):
        super(up, self).__init__()
        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True
        )
        self.conv = double_conv(in_ch, out_ch, kernel_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
