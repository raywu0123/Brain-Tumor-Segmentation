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
        use_position=False,
        dropout_rate: int = 0.,
        attention: bool = False,
        self_attention: int = 0,
        **kwargs,
    ):
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.conv_times = conv_times
        self.use_position = use_position
        super(UNet, self).__init__(
            batch_sampler_id=batch_sampler_id,
            data_format=data_format,
            forward_outcome_channels=channel_num,
            head_outcome_channels=channel_num,
            **kwargs,
        )
        self.floor_num = floor_num
        image_chns = data_format['channels']
        if use_position:
            image_chns += 1
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        self.self_attention_module = nn.Sequential(*[
          SelfAttention(in_dim=2 ** floor_num * channel_num)
          for _ in range(self_attention)
        ])

        for floor_idx in range(floor_num):
            channel_times = 2 ** floor_idx
            d = DownConv(channel_num * channel_times, kernel_size, conv_times, self.dropout_rate)
            self.down_layers.append(d)

        for floor_idx in range(floor_num)[::-1]:
            channel_times = 2 ** floor_idx
            up_conv_class = AttentionUpConv if attention else UpConv
            u = up_conv_class(
                channel_num * 2 * channel_times,
                kernel_size,
                conv_times,
                self.dropout_rate,
            )
            self.up_layers.append(u)

    def forward_head(self, inp, data_idx):
        inp, pos = inp['slice'], inp['position']
        x = normalize_batch_image(inp)
        x = get_tensor_from_array(x)

        if self.use_position:
            pos = get_tensor_from_array(pos)
            pos = pos.view(pos.shape[0], 1, 1, 1)
            pos = pos.expand(-1, 1, x.shape[-2], x.shape[-1])
            x = torch.cat([x, pos], dim=1)

        x = self.heads[data_idx](x)
        return x

    def forward(self, x):
        x_out = [x]
        for down_layer in self.down_layers:
            x = down_layer(x)
            x_out.append(x)

        x = self.self_attention_module(x)
        x_out = x_out[:-1]
        for x_down, u in zip(x_out[::-1], self.up_layers):
            x = u(x, x_down)
        return x

    def build_heads(self, input_channels: list, output_channel: int):
        if self.use_position:
            input_channels = [chn + 1 for chn in input_channels]
        return nn.ModuleList([
            ConvNTimes(
                input_channel,
                output_channel,
                self.kernel_size,
                self.conv_times,
                self.dropout_rate,
            )
            for input_channel in input_channels
        ])

    def build_tails(self, input_channels, class_nums):
        return nn.ModuleList([
            nn.Sequential(
                ConvNTimes(
                    input_channels,
                    input_channels,
                    self.kernel_size,
                    self.conv_times,
                    self.dropout_rate,
                ),
                nn.Conv2d(input_channels, class_num, kernel_size=1)
            )
            for class_num in class_nums
        ])


class ConvNTimes(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size, conv_times, dropout_rate):
        super(ConvNTimes, self).__init__()
        assert(conv_times > 0)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.conv_times = conv_times

        self.dropout = nn.Dropout2d(p=dropout_rate)

        self.in_conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        for _ in range(conv_times - 1):
            conv = nn.Conv2d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
            self.convs.append(conv)
        for _ in range(conv_times):
            norm = nn.InstanceNorm2d(out_ch)
            self.norms.append(norm)

    def forward(self, inp):
        inp = self.in_conv(inp)
        x = inp
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = self.dropout(x)
            if self.dropout.p == 0.:
                x = norm(x)
            x = F.relu(x)

        x = (x + inp) / 2
        return x


class DownConv(nn.Module):

    def __init__(self, in_ch, kernel_size, conv_times, dropout_rate):
        out_ch = in_ch * 2
        super(DownConv, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvNTimes(in_ch, out_ch, kernel_size, conv_times, dropout_rate)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class UpConv(nn.Module):

    def __init__(self, in_ch, kernel_size, conv_times, dropout_rate):
        super(UpConv, self).__init__()
        out_ch = in_ch // 2
        self.conv_transpose = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size,
            padding=kernel_size // 2,
            stride=2,
        )
        self.conv = ConvNTimes(in_ch, out_ch, kernel_size, conv_times, dropout_rate)

    def forward(self, x_down, x_up):
        x_down = self.conv_transpose(x_down)
        diff_x = x_up.size()[2] - x_down.size()[2]
        diff_y = x_up.size()[3] - x_down.size()[3]
        x_down = F.pad(x_down, [0, diff_x, 0, diff_y])
        x = torch.cat([x_down, x_up], dim=1)
        x = self.conv(x)
        return x


class AttentionUpConv(nn.Module):

    def __init__(self, in_ch, kernel_size, conv_times, dropout_rate):
        super().__init__()
        out_ch = in_ch // 2
        self.conv_transpose = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size,
            padding=kernel_size // 2,
            stride=2,
        )
        self.query_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=1,
        )
        self.key_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=1,
        )
        self.attention_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=1,
            kernel_size=1,
        )
        self.n_conv = ConvNTimes(
            in_ch, out_ch, kernel_size, conv_times, dropout_rate
        )

    def forward(self, x, x_down):
        x = self.conv_transpose(x)
        diff_x = x_down.size()[2] - x.size()[2]
        diff_y = x_down.size()[3] - x.size()[3]
        x = F.pad(x, [0, diff_x, 0, diff_y])

        q = self.query_conv(x)
        k = self.key_conv(x_down)
        attention = torch.sigmoid(self.attention_conv(F.relu(q + k)))
        value = x_down * attention
        concat = torch.cat([x, value], dim=1)
        out = self.n_conv(concat)
        return out


class SelfAttention(nn.Module):
    """
    Self attention Layer, adapted from
    https://github.com/heykeetae/Self-Attention-GAN/blob/master/sagan_models.py
    """

    def __init__(self, in_dim):
        super().__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize,
            -1,
            width * height
        ).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out
