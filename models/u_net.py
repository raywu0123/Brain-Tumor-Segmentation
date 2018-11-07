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
            class_num: int = 2,
            lr: float = 1e-4,
            kernel_size: int = 3,
            floor_num: int = 4,
            channel_num: int = 64,
            conv_times: int = 2,
        ):
        super(UNet, self).__init__(
            channels=channels,
            depth=depth,
            height=height,
            width=width,
            metadata_dim=metadata_dim,
            class_num=class_num,
        )
        self.model = UNet_Net(
            image_chns=self.data_channels,
            kernel_size=kernel_size,
            floor_num=floor_num,
            channel_num=channel_num,
            conv_times=conv_times,
            class_num=class_num,
        )
        self.opt = optim.Adam(params=self.model.parameters(), lr=lr)
        if torch.cuda.is_available():
            self.model.cuda()


class UNet_Net(nn.Module):
    def __init__(
        self,
        image_chns,
        floor_num,
        kernel_size,
        channel_num,
        conv_times,
        class_num,
    ):
        super(UNet_Net, self).__init__()
        self.floor_num = floor_num
        self.image_chns = image_chns
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        in_conv = Conv_N_Times(image_chns, channel_num, kernel_size, conv_times)
        self.down_layers.append(in_conv)
        for floor_idx in range(floor_num):
            channel_times = 2 ** floor_idx
            d = down(channel_num * channel_times, kernel_size, conv_times)
            self.down_layers.append(d)

        for floor_idx in range(floor_num)[::-1]:
            channel_times = 2 ** floor_idx
            u = up(channel_num * 2 * channel_times, kernel_size, conv_times)
            self.up_layers.append(u)
        self.out_conv = nn.Conv2d(channel_num, class_num, kernel_size=1)

    def forward(self, x):
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


class Conv_N_Times(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, conv_times):
        super(Conv_N_Times, self).__init__()
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


class down(nn.Module):
    def __init__(self, in_ch, kernel_size, conv_times):
        out_ch = in_ch * 2
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            Conv_N_Times(in_ch, out_ch, kernel_size, conv_times)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, kernel_size, conv_times, bilinear=True):
        super(up, self).__init__()
        out_ch = in_ch // 2
        self.conv_transpose = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size,
            padding=kernel_size // 2,
            stride=2,
        )
        self.conv = Conv_N_Times(in_ch, out_ch, kernel_size, conv_times)

    def forward(self, x_down, x_up):
        # x1 = self.up(x1)
        x_down = self.conv_transpose(x_down)
        diffX = x_up.size()[2] - x_down.size()[2]
        diffY = x_up.size()[3] - x_down.size()[3]
        x_down = F.pad(x_down, (0, diffX, 0, diffY))
        x = torch.cat([x_down, x_up], dim=1)
        x = self.conv(x)
        return x
