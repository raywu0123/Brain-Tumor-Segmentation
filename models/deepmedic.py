import torch
import torch.nn as nn
import torch.nn.functional as F


# from .base import PytorchModelBase
# from .loss_functions import weighted_cross_entropy
# from .utils import get_tensor_from_array
###########################################################
#                  Deepmedic                              #
#  input    [batch_num, in_channel,   D1, H1,  W1 ]       #
#           [batch_num, in_channel,   D2, H2,  W2 ]       #
#  to work properly, D1, H1, W1 should be bigger than     #
#  D2, H2, W2                                             #
#                                                         #
#  output   [batch_num, out_channel,  D', H',  W' ]       #
#  D' = D1 - (kernel_size - 1) * conv_time * block_num    #
#  H' = H1 - (kernel_size - 1) * conv_time * block_num    #
#  W' = W1 - (kernel_size - 1) * conv_time * block_num    #
#  notice that D', H', W' should > 0                      #
#                                                         #
#                 for initialize                          #
#  data_format = [x1.shape, x2.shape]                     #
#  channel_list =                                         #
#  [[x0, x1], [x1, x2] ... [xn-3, xn-2], [xn-1, xn]]      #
#   x0      = input_channel                               #
#   x1      = duplicate_num                               #
#   x2~xn-2 = block channel num                           #
#   xn-1    = final block input channel, = 2 * xn-2       #
###########################################################
class Deepmedic(nn.Module):
    def __init__(
            self,
            data_format: dict,
            channel_list: dict,
            out_channel: int = 2,
            kernel_size: int = 3,
            conv_time: int = 2,
            batch_sampler_id='three_dim',
        ):
        super(Deepmedic, self).__init__()
        if kernel_size % 2 == 0:
            raise AssertionError('kernel_size({}) must be odd'.format(kernel_size))

        self.duplicate1 = Duplicate(channel_list[0][0], channel_list[0][1], kernel_size)
        self.duplicate2 = Duplicate(channel_list[0][0], channel_list[0][1], kernel_size)

        self.first = Path(channel_list[1:-1], kernel_size, conv_time)
        self.second = Path(channel_list[1:-1], kernel_size, conv_time)
        self.block = Block(channel_list[-1][0], channel_list[-1][1], kernel_size, conv_time)

        self.out = Block(channel_list[-1][1], out_channel, kernel_size=1, conv_time=1, route=False)

    def forward(self, x1, x2):
        x1 = get_tensor_from_array(x1)
        x2 = get_tensor_from_array(x2)

        x1 = self.duplicate1(x1)
        x2 = self.duplicate2(x2)

        x1 = self.first(x1)
        x2 = self.second(x2)

        if x1.shape[2] < x2.shape[2]:
            raise AssertionError('Shape error, low resolution{} must be smaller than \
                                 high resolution{}'.format(x2.shape, x1.shape))
        """
        upsampling here can be changed to convtranspose3d
        """
        x2 = F.interpolate(x2, x1.shape[2:])

        x = torch.cat((x1, x2), 1)
        x = self.block(x)

        x = self.out(x)
        return x


class Path(nn.Module):
    def __init__(self, channel_list, kernel_size, conv_time):
        super(Path, self).__init__()
        self.block_list = nn.ModuleList()
        channel = channel_list[0]
        block = Block(channel[0], channel[1], kernel_size, conv_time, route=False)
        self.block_list.append(block)
        for channel in channel_list[1:]:
            block = Block(channel[0], channel[1], kernel_size, conv_time)
            self.block_list.append(block)

    def forward(self, x):
        for block in self.block_list:
            x = block(x)
        return x


###########################################################
#              Block                                      #
#  input    [batch_num, in_channel,     D,  H,  W ]       #
#           route = True for residual connection          #
#  output   [batch_num, out_channel*2,  D', H', W']       #
#  D' = D - (kernel_size - 1) * conv_time                 #
#  H' = H - (kernel_size - 1) * conv_time                 #
#  W' = W - (kernel_size - 1) * conv_time                 #
###########################################################
class Block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, conv_time=2, route=True):
        super(Block, self).__init__()
        self.route = route

        self.norm_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()

        self.activation = nn.ReLU()

        batch_norm = nn.BatchNorm3d(input_channel)
        self.norm_list.append(batch_norm)

        conv = nn.Conv3d(input_channel, output_channel, kernel_size=kernel_size)
        self.conv_list.append(conv)

        for i in range(conv_time - 1):
            batch_norm = nn.BatchNorm3d(output_channel)
            self.norm_list.append(batch_norm)

            conv = nn.Conv3d(output_channel, output_channel, kernel_size=kernel_size)
            self.conv_list.append(conv)

    def forward(self, x):
        x1 = x
        for (norm, conv) in zip(self.norm_list, self.conv_list):
            """
            the order of norm, active, conv is weird
            """
            x1 = norm(x1)
            x1 = self.activation(x1)
            x1 = conv(x1)

        if self.route:
            # crop x to x1
            d_ch = x1.shape[1] - x.shape[1]
            d_pix = x.shape[2] - x1.shape[2]
            if d_pix % 2 != 0:
                raise AssertionError('shape Error', x.shape, x1.shape)
            d = d_pix // 2
            x = x[:, :, d:-d, d:-d, d:-d]

            if d_ch != 0:
                # add channel to x
                b = x.shape[0]
                d = x.shape[2]
                h = x.shape[3]
                w = x.shape[4]
                empty = torch.zeros((b, d_ch, d, h, w))
                if x.is_cuda:
                    empty = empty.cuda()
                x = torch.cat((x, empty), 1)
            x1 = x + x1
        return x1


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
        self.batch_norm = nn.BatchNorm3d(duplication_num)

    def forward(self, inp):
        x = self.duplicate(inp)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x


# ---------- testing function ---------- #
def get_tensor_from_array(x):
    x = torch.from_numpy(x).float().cuda()
    return x


if __name__ == '__main__':
    import numpy as np
    x1 = np.zeros((10, 1, 25, 25, 25))
    x2 = np.zeros((10, 1, 19, 19, 19))
    data_format = np.array([x1.shape, x2.shape])
    channel_list = [[1, 4], [4, 30], [30, 40], [40, 40], [40, 50], [50 * 2, 150]]
    deepmedic = Deepmedic(data_format, channel_list).cuda()
    x = deepmedic(x1, x2)
    print(x.shape)
