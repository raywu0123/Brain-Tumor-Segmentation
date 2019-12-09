from torch import nn

from .base import PytorchModelBase
from .utils import get_tensor_from_array


class HighResolutionCompactNetwork(PytorchModelBase):

    def __init__(
            self,
            data_format: dict,
            batch_sampler_id='three_dim',
            duplication_num=32,
            kernel_size=3,
            interpolate_size=(96, 96, 96),
            **kwargs,
    ):
        self.kernel_size = kernel_size
        self.data_format = data_format
        self.interpolate_size = interpolate_size
        super().__init__(
            batch_sampler_id=batch_sampler_id,
            data_format=data_format,
            head_outcome_channels=duplication_num,
            forward_outcome_channels=duplication_num,
            **kwargs,
        )
        self.seq_layers = nn.Sequential(
            CustomConv(duplication_num, duplication_num),
            CustomConv(duplication_num, duplication_num),
            CustomConv(duplication_num, duplication_num),
            CustomConv(duplication_num, duplication_num, dilation=2),
            CustomConv(duplication_num, duplication_num, dilation=2),
            CustomConv(duplication_num, duplication_num, dilation=2),
            CustomConv(duplication_num, duplication_num, dilation=4),
            CustomConv(duplication_num, duplication_num, dilation=4),
            CustomConv(duplication_num, duplication_num, dilation=4),
        )

    def forward_head(self, inp, data_idx):
        x = get_tensor_from_array(inp)
        if x.dim() != 5:
            raise AssertionError('input must have shape (batch_size, channel, D, H, W),\
                                 but get {}'.format(x.shape))
        return self.heads[data_idx](x)

    def build_heads(self, input_channels: list, output_channel: int):
        return nn.ModuleList([
            nn.Sequential(
                Interpolate(size=self.interpolate_size),
                nn.Conv3d(
                    input_channel,
                    output_channel,
                    self.kernel_size,
                    padding=self.kernel_size // 2,
                ),
                nn.BatchNorm3d(output_channel),
                nn.ReLU(),
            )
            for input_channel in input_channels
        ])

    def forward(self, x):
        x = self.seq_layers(x)
        return x

    def build_tails(self, input_channels, class_nums):
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(input_channels, class_num, kernel_size=1),
                Interpolate(
                    size=(
                        data_format['depth'],
                        data_format['height'],
                        data_format['width'],
                    )
                )
            )
            for class_num, data_format in zip(class_nums, self.all_data_formats)
        ])


class CustomConv(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            dilation=1,
            kernel_size=3,
            repeats=2,
            batch_norm=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm3d(in_channels) if batch_norm else nn.Sequential(),
                nn.ReLU(),
                nn.Conv3d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    dilation=dilation,
                    padding=(dilation * (kernel_size - 1)) // 2,
                ),
            )
            for _ in range(repeats)
        ])

    def forward(self, inp):
        x = inp
        for layer in self.layers:
            x = layer(x)
        return x + inp


class Interpolate(nn.Module):

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size)
