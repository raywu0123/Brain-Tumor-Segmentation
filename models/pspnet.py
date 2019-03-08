# Adapted From https://github.com/Lextal/pspnet-pytorch

import torch
from torch import nn
from torch.nn import functional as F

from .base import PytorchModelBase
from .utils import get_tensor_from_array
from .loss_functions import ce_minus_log_dice
from .extractors import extractor_hub


class PSPNet(PytorchModelBase):

    def __init__(
            self,
            data_format: dict,
            sizes=(1, 2, 3, 6, 10),
            psp_size=512,
            # deep_features_size=256,
            backend='resnet34',
            pretrained=False,
            batch_sampler_id='two_dim',
    ):
        super().__init__(
            batch_sampler_id=batch_sampler_id,
            loss_fn=ce_minus_log_dice,
            data_format=data_format,
        )
        self.feats = extractor_hub[backend](data_format['channels'], pretrained)
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)

        n_classes = data_format['class_num']
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Softmax(dim=1),
        )

        # self.classifier = nn.Sequential(
        #     nn.Linear(deep_features_size, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, n_classes)
        # )

    def forward(self, x):
        x = get_tensor_from_array(x)

        f, class_f = self.feats(x)
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        # auxiliary = F.adaptive_max_pool2d(
        #     input=class_f,
        #     output_size=(1, 1)
        # ).view(-1, class_f.size(1))
        p = self.final(p)
        return p  # , self.classifier(auxiliary)


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
