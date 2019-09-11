# Adapted From https://github.com/Lextal/pspnet-pytorch

import torch
from torch import nn
from torch.nn import functional as F

from .base import PytorchModelBase
from .utils import get_tensor_from_array
from .extractors import extractor_hub


class PSPNet(PytorchModelBase):

    def __init__(
            self,
            data_format: dict,
            batch_sampler_id='two_dim',
            sizes=(1, 2, 3, 6, 10),
            psp_size=512,
            backend='resnet34',
            pretrained=False,
            **kwargs,
    ):
        super().__init__(
            batch_sampler_id=batch_sampler_id,
            data_format=data_format,
            head_outcome_channels=data_format['channels'],
            forward_outcome_channels=64,
            **kwargs,
        )
        self.feats = extractor_hub[backend](data_format['channels'], pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)

    def forward_head(self, inp, data_idx: int):
        x = inp['slice']
        x = get_tensor_from_array(x)
        x = self.heads[data_idx](x)
        return x

    def forward(self, x):
        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        p = self.final(p)
        return p

    def build_heads(self, input_channels: list, output_channel: int):
        return nn.ModuleList([
            nn.Conv2d(input_channel, output_channel, kernel_size=3)
            for input_channel in input_channels
        ])

    def build_tails(self, input_channels, class_nums):
        return nn.ModuleList([
            nn.Conv2d(input_channels, class_num, kernel_size=1)
            for class_num in class_nums
        ])


class PSPModule(nn.Module):

    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(
                input=stage(feats),
                size=(h, w),
                mode='bilinear',
                align_corners=True,
            ) for stage in self.stages
        ] + [feats]

        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class PSPUpsample(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=True)
        return self.conv(p)
