import torch
import torch.nn as nn
import torch.nn.functional as F
from dotenv import load_dotenv
load_dotenv('./.env')
import torch.optim as optim

from .base import Model2DBase


class UNet(Model2DBase):
    def __init__(
            self,
            channels: int = 1,
            depth: int = 200,
            height: int = 200,
            width: int = 200,
            metadata_dim: int = 0,
            lr: float = 1e-4,
            kernel_size: int = 3,
            floor_num: int = 2,
            channel_num: int = 48,
            conv_times: int = 2,
        ):
        super(UNet, self).__init__()
        self.model = UNet_structure(
            image_chns=self.data_channels,
            kernel_size=kernel_size,
            floor_num=floor_num,
            channel_num=channel_num,
            conv_times=conv_times,
        )
        self.opt = optim.Adam(params=self.model.parameters(), lr=lr)
        if torch.cuda.is_available():
            self.model.cuda()


class UNet_structure(nn.Module):
    def __init__(
        self,
        image_chns,
        floor_num,
        kernel_size,
        channel_num,
        conv_times,
        class_num=1,
    ):
        super(UNet_structure, self).__init__()
        self.floor_num = floor_num
        self.image_chns = image_chns
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        self.inc = inconv(image_chns, channel_num, kernel_size, conv_times)
        self.outc = outconv(channel_num, class_num)

        for floor in range(floor_num):
            channel_times = 2 ** floor
            d = down(channel_num * channel_times, kernel_size, conv_times)
            self.down_layers.append(d)

        for floor in range(floor_num):
            f_b = floor_num - floor - 1
            channel_times = 2 ** f_b
            u = up(channel_num * 2 * channel_times, kernel_size, conv_times)
            self.up_layers.append(u)

    def forward(self, x):
        x_out = []
        x = self.inc(x)
        x_out.append(x)
        for d in (self.down_layers):
            x = d(x)
            x_out.append(x)
        x_out = x_out[:-1]
        for x_down, u in zip(x_out[::-1], self.up_layers):
            x = u(x, x_down)
        x = self.outc(x)
        x = torch.sigmoid(x)
        return x


class conv_n_times(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, conv_times):
        super(conv_n_times, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.batchnorm = nn.BatchNorm2d(out_ch)
        self.conv_times = conv_times

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = F.relu(x)
        for _ in range(self.conv_times - 1):
            x = self.conv2(x)
            x = self.batchnorm(x)
            x = F.relu(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, conv_times):
        super(inconv, self).__init__()
        self.conv = conv_n_times(in_ch, out_ch, kernel_size, conv_times)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, kernel_size, conv_times):
        out_ch = in_ch * 2
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            conv_n_times(in_ch, out_ch, kernel_size, conv_times)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, kernel_size, conv_times, bilinear=True):
        super(up, self).__init__()
        out_ch = in_ch // 2
        self.convtran = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.up = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True
        )
        self.conv = conv_n_times(in_ch, out_ch, kernel_size, conv_times)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.convtran(x1)
        diffX = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (0, diffX, 0, diffY))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
