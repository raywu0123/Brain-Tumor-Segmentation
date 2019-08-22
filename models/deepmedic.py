import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import ceil

from .base import PytorchModelBase
from .loss_functions import ce_minus_log_dice_with_size_mismatch
from .utils import get_tensor_from_array


###########################################################
#                  Deepmedic                              #
#  input    [batch_num, in_channel,   D,  H,   W  ]       #
#  [D1, H1, W1] =  [D, H, W] // 2                         #
#                                                         #
#  output   [batch_num, out_channel,  D', H',  W' ]       #
#  D' = D1 - (kernel_size - 1) * conv_time * block_num    #
#  H' = H1 - (kernel_size - 1) * conv_time * block_num    #
#  W' = W1 - (kernel_size - 1) * conv_time * block_num    #
#  notice that D', H', W' should > 0                      #
#                                                         #
#                 for initialize                          #
#  channel_list =                                         #
#  [[x0, x1], [x1, x2] ... [xn-3, xn-2], [xn-1, xn]]      #
#   x0      = input_channel                               #
#   x1      = duplicate_num                               #
#   x2~xn-2 = block channel num                           #
#   xn-1    = final block input channel, = 2 * xn-2       #
###########################################################
class Deepmedic(PytorchModelBase):
    def __init__(
            self,
            data_format: dict,
            channel_list: list,
            kernel_size: int = 3,
            conv_time: int = 2,
            dim_in: int = 64,
            batch_sampler_id='center_patch3d',
        ):
        super(Deepmedic, self).__init__(
            data_format=data_format,
            batch_sampler_id=batch_sampler_id,
            loss_fn=ce_minus_log_dice_with_size_mismatch,
        )
        if kernel_size % 2 == 0:
            raise AssertionError('kernel_size({}) must be odd'.format(kernel_size))
        self.dim_in = dim_in
        self.pool = nn.AvgPool3d(kernel_size=5, stride=3, padding=1)

        self.duplicate1 = Duplicate(data_format['channels'], channel_list[0][1], kernel_size)
        self.duplicate2 = Duplicate(data_format['channels'], channel_list[0][1], kernel_size)

        self.first = Path(channel_list[1:-1], kernel_size, conv_time)
        self.second = Path(channel_list[1:-1], kernel_size, conv_time)
        self.block = Block(channel_list[-1][0], channel_list[-1][1], 1, conv_time)

        self.out = Block(channel_list[-1][1], data_format['class_num'],
                         kernel_size=1, conv_time=1, route=False)

    def forward(self, inp):
        inp = get_tensor_from_array(inp)

        # x1 : for high resolution, x2 for low resolution
        # c_ : cut inp to smaller piece in the middle
        c_d = inp.shape[2] // 4
        c_h = inp.shape[3] // 4
        c_w = inp.shape[4] // 4
        x1 = inp[:, :, c_d:-c_d, c_h:-c_h, c_w:-c_w]
        x2 = self.pool(inp)

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
        x = F.softmax(x, dim=1)
        return x

    def predict(self, test_data, **kwargs):
        test_data = test_data['volume']
        self.eval()
        dim_in = 64
        dim_out = 16
        patch_list = get_patch(test_data, dim_in, dim_out)
        pred_list = []
        for patch in patch_list:
            pred = self.forward(patch).cpu().data.numpy()
            pred_list.append(pred)
        assert(patch_list[0].shape[-1] == dim_in)
        assert(pred_list[0].shape[-1] == dim_out)
        return reassemble(pred_list, test_data, dim_out)


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
#           [route] True for residual connection          #
#                                                         #
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

            if d_pix != 0:
                if d_pix % 2 != 0:
                    raise AssertionError('shape Error', x.shape, x1.shape)
                d = d_pix // 2
                x = x[:, :, d:-d, d:-d, d:-d]

            if d_ch != 0:
                # add channel to x
                pad_shape = list(x.shape)
                pad_shape[1] = d_ch
                empty = torch.zeros(pad_shape)
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

    def forward(self, x):
        x = self.duplicate(x)
        return x


"""
functions for deepmedic prediction

function description

having a function
inp = [N, c_in,  64, 64, 64]
out = [N, c_out, 16, 16, 16]

we want to predict array [N, c_out, d, h, w]
"""


def get_patch(data, dim_in, dim_out):
    # first, we need to pad the array with size (dim_in - dim_out) / 2 arround
    # since our ouput will loss such pixels at each boundary

    if (dim_in - dim_out) % 2 != 0:
        raise AssertionError("this should be even")
    pad = (dim_in - dim_out) // 2

    data_pad = np.pad(data,
                      ((0, 0), (0, 0), (pad, pad), (pad, pad), (pad, pad)),
                      'constant',
                      constant_values=0)

    # pad data such that it can be divide by dim_out
    if data.shape[2] % dim_out != 0:
        d_p = dim_out - data.shape[2] % dim_out
    else:
        d_p = 0

    if data.shape[3] % dim_out != 0:
        h_p = dim_out - data.shape[3] % dim_out
    else:
        h_p = 0

    if data.shape[4] % dim_out != 0:
        w_p = dim_out - data.shape[4] % dim_out
    else:
        w_p = 0

    data_pad = np.pad(data_pad,
                      ((0, 0), (0, 0), (0, d_p), (0, h_p), (0, w_p)),
                      'constant',
                      constant_values=0)

    # then we need to get the vertex of patch
    vertex_list = []
    for d in range(data_pad.shape[2] // dim_out):
        for h in range(data_pad.shape[3] // dim_out):
            for w in range(data_pad.shape[4] // dim_out):
                vertex_list.append([d * dim_out, h * dim_out, w * dim_out])

    patch_list = []
    shape = data_pad.shape
    for vertex in vertex_list:
        d_e = vertex[0] + dim_in
        h_e = vertex[1] + dim_in
        w_e = vertex[2] + dim_in
        if(d_e <= shape[2] and h_e <= shape[3] and w_e <= shape[4]):
            patch = data_pad[:, :,
                             vertex[0]:vertex[0] + dim_in,
                             vertex[1]:vertex[1] + dim_in,
                             vertex[2]:vertex[2] + dim_in]
            patch_list.append(patch)
    return patch_list


def reassemble(pred_list, data, dim_out):
    d_n = ceil(data.shape[2] / dim_out)
    h_n = ceil(data.shape[3] / dim_out)
    w_n = ceil(data.shape[4] / dim_out)

    # this assertion should never occur, or the logit is wrong
    # to fix it you might rewrite the prediction
    if len(pred_list) != (d_n * h_n * w_n):
        raise AssertionError("shape mismatch", len(pred_list),
                             d_n * h_n * w_n)

    shape = list(data.shape)
    shape[1] = pred_list[0].shape[1]
    pred_all = np.zeros(shape)
    idx = 0
    for d in range(d_n):
        for h in range(h_n):
            for w in range(w_n):
                pred = pred_list[idx]
                # consider the situation at the boundary
                d_cut = dim_out
                if d * dim_out + dim_out > data.shape[2]:
                    d_cut = data.shape[2] - d * dim_out

                h_cut = dim_out
                if h * dim_out + dim_out > data.shape[3]:
                    h_cut = data.shape[3] - h * dim_out

                w_cut = dim_out
                if w * dim_out + dim_out > data.shape[4]:
                    w_cut = data.shape[4] - w * dim_out

                pred_all[
                    :, :,
                    d * dim_out:d * dim_out + d_cut,
                    h * dim_out:h * dim_out + h_cut,
                    w * dim_out:w * dim_out + w_cut,
                ] = pred[:, :, :d_cut, :h_cut, :w_cut]
                idx = idx + 1
    return pred_all


# -------- testing functions for prediction -------- #
if __name__ == "__main__":
    data = np.random.rand(1, 2, 200, 200, 200)
    patch_list = get_patch(data, 64, 32)
    pred_list = []
    for patch in patch_list:
        # just test this by putting your output shape below
        pred = patch[:, :, 16:-16, 16:-16, 16:-16]
        pred_list.append(pred)
    pred_all = reassemble(pred_list, data, 32)
    print(np.all(pred_all == data))
