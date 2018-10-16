import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import Model2DBase
from dotenv import load_dotenv
import torch.optim as optim
load_dotenv('./.env')


class UNet(Model2DBase):
    def __init__(
            self,
            channels: int = 1,
            depth: int = 200,
            height: int = 200,
            width: int = 200,
            metadata_dim: int = 0,
            num_units: [int] = (32, 32, 64, 64, 128),
            pooling_layer_num: [int] = (1, 3),
            kernel_size: int = 3,
            lr: float = 1e-4,
        ):
        super(UNet, self).__init__(
            channels=channels,
            depth=depth,
            height=height,
            width=width,
            metadata_dim=metadata_dim,
        )
        self.model = UNet_structure(
            image_chns=self.data_channels,
            kernel_size=kernel_size,
            floor_num=2,
            channel_num=48,
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
    ):
        super(UNet_structure, self).__init__()
        self.floor_num = floor_num
        self.image_chns = image_chns
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        self.inc = inconv(image_chns, channel_num, kernel_size)
        for floor in range(floor_num):
            a = 2 ** floor
            d = down(channel_num * a, channel_num * a * 2, kernel_size)
            self.down_layers.append(d)
        a = 2 ** floor_num
        self.down_end = down(channel_num * a, channel_num * a, kernel_size)

        for floor in range(floor_num):
            f_b = floor_num - floor - 1
            a = 2 ** f_b
            u = up(channel_num * 4 * a, channel_num * a, kernel_size)
            self.up_layers.append(u)
        self.up_end = up(channel_num * 2, channel_num * 1, kernel_size)
        self.outc = outconv(channel_num, image_chns)

    def forward(self, x):
        x_out = []
        x = self.inc(x)
        x_out.append(x)
        for d in (self.down_layers):
            x = d(x)
            x_out.append(x)
        x = self.down_end(x)
        for floor, u in enumerate(self.up_layers):
            f_b = self.floor_num - floor
            x = u(x, x_out[f_b])
        x = self.up_end(x, x_out[0])
        x = self.outc(x)
        return x


<<<<<<< HEAD
=======
<<<<<<< HEAD
class conv_n_times(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, conv_times):
        super(conv_n_times, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.batchnorm = nn.BatchNorm2d(out_ch)
        self.conv_times = conv_times
=======
>>>>>>> 088f52d... UNet
class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
<<<<<<< HEAD
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
=======
            nn.Conv2d(in_ch, out_ch, kernel_size, padding="same"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding="same"),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
>>>>>>> f5f557a... fix
>>>>>>> 088f52d... UNet

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
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
