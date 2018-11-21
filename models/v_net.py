import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from .utils import weighted_cross_entropy, soft_dice_score


def Activation():
    return nn.ReLU()


class VNet(nn.Module):
    def __init__(
            self,
            channels: int = 1,
            depth: int = 200,
            height: int = 200,
            width: int = 200,
            metadata_dim: int = 0,
            class_num: int = 2,
            lr: float = 1e-4,
        ):
        super(VNet, self).__init__()
        self.model = Vnet_net().cuda()
        self.opt = optim.Adam(params=self.model.parameters(), lr=lr)
        
    def fit_dataloader(self, get_training_dataloader, get_validation_dataloader, **kwargs):
        batch_size = 1
        training_dataloader = get_training_dataloader(batch_size, shuffle=True, num_workers=0)
        for i_batch, sampled_batch in enumerate(training_dataloader):
            self.opt.zero_grad()
            # sampled_batch['volume']
            data = sampled_batch['volume'].reshape(1, 1, 200, 200, 200).cuda().float()
            label = sampled_batch['label'].reshape(1, 2, 200, 200, 200).cuda().float()
            pre = self.model(data)
            crossentropy_loss = weighted_cross_entropy(pre, label)
            dice_score = soft_dice_score(pre, label)
            total_loss = crossentropy_loss - torch.log(dice_score)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            total_loss.backward()
            self.opt.step()
            print(dice_score)
        


###########################################################
#             Vnet_net                                    #
#  input   [batch_num, input_channel,  D,   H,   W]       #
#  output  [batch_num,             2,  D,   H,   W]       #
###########################################################
class Vnet_net(nn.Module):
    def __init__(self, 
                 input_channel=1,
                 duplication_num=8,
                 kernel_size=5,
                 conv_time=2,
                 n_layer=4
                 ):
        super(Vnet_net, self).__init__()
        # To work properly, kernel_size must be odd
        if (kernel_size % 2 == 0):
            raise AssertionError('kernel_size({}) must be odd'.format(kernel_size))
        self.n_layer = n_layer
        # define list
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        # layer
        self.duplicate = Duplicate(input_channel, duplication_num, kernel_size)
        for i in range(n_layer):
            n_channel = np.power(2, i) * duplication_num
            dnConv = DnConv(n_channel, kernel_size, conv_time)
            self.down.append(dnConv)
        for i in range(n_layer - 1):
            n_channel = np.power(2, i) * duplication_num
            upConv = UpConv(n_channel * 4, n_channel, kernel_size, conv_time)
            self.up.append(upConv)
        n_channel = np.power(2, n_layer - 1) * duplication_num
        upConv = UpConv(n_channel * 2, n_channel, kernel_size, conv_time)
        self.up.append(upConv)
        self.output_layer = Out_layer(duplication_num * 2)

    def forward(self, inp):
        # check
        if(inp.dim() != 5):
            raise AssertionError('input must have shape (batch_size, channel, D, H, W)')
#         check_shape(inp.shape, self.n_layer)
        # start
        x_out = []
        # turn input channel to duplication_num
        x = self.duplicate(inp)
        x_out.append(x)
        # down conv
        for i in range(self.n_layer):
            x = self.down[i](x)
            x_out.append(x)
        # up conv
        for i in range(self.n_layer):
            n_up = self.n_layer - i - 1
            x = self.up[n_up](x, x_out[n_up])
        # out_layer
        x = self.output_layer(x)
        return x


###########################################################
#             DnConv                                      #
#  input   [batch_num, input_channel,   D,   H,   W]      #
#  output  [batch_num, output_channel,  D/2, H/2, W/2]    #
###########################################################
class DnConv(nn.Module):
    def __init__(self, input_channel, kernel_size, conv_time):
        super(DnConv, self).__init__()
        output_channel = input_channel * 2
        self.dnconv = nn.Conv3d(input_channel, output_channel, kernel_size=2, stride=2)
        self.activation = Activation()
        self.batch_norm = nn.BatchNorm3d(output_channel)
        self.conv_N_time = Conv_N_time(output_channel, kernel_size, conv_time)

    def forward(self, x):
        x = self.dnconv(x)
        x = self.activation(x)
        x = self.batch_norm(x)
        x = self.conv_N_time(x)
        return x


###########################################################
#             UpConv                                      #
#  x1      [batch_num, x1_channel,    D/2, H/2, W/2]      #
#  x2      [batch_num, x2_channel,    D,   H,   W]        #
#  output  [batch_num, x2_channel*2,  D*2, H*2, W*2]      #
###########################################################
class UpConv(nn.Module):
    def __init__(self, x1_channel, x2_channel, kernel_size, conv_time):
        super(UpConv, self).__init__()
        self.upconv = nn.ConvTranspose3d(x1_channel, x2_channel, kernel_size=2, stride=2)
        self.activation = Activation()
        self.batch_norm = nn.BatchNorm3d(x2_channel)
        self.conv_N_time = Conv_N_time(x2_channel * 2, kernel_size, conv_time)
        self.padding = nn.ConstantPad3d((0, 1, 0, 1, 0, 1), 0)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x1 = self.activation(x1)
        x1 = self.batch_norm(x1)
        if(x1.shape != x2.shape):
            # this case will only happend for 
            # x1 [N, C, D-1, H-1, W-1]
            # x2 [N, C, D,   H,   W  ]
            x1 = self.padding(x1)
        x = torch.cat((x1, x2), 1)
        x = self.conv_N_time(x)
        return x


###########################################################
#             Conv_N_time                                 #
#  input   [batch_num, channel_num,   D,   H,   W]        #
#  output  [batch_num, channel_num,   D,   H,   W]        #
###########################################################
class Conv_N_time(nn.Module):
    def __init__(self, channel_num, kernel_size, N):
        super(Conv_N_time, self).__init__()
        # define list
        self.convs = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.conv_times = N
        # define layers
        self.activation = Activation()
        for _ in range(N):
            conv = nn.Conv3d(channel_num, channel_num, kernel_size=kernel_size, padding=kernel_size//2)
            self.convs.append(conv)
        for _ in range(N):
            norm = nn.BatchNorm3d(channel_num)
            self.batchnorms.append(norm)
    
    def forward(self, x):
        for conv, batchnorm in zip(self.convs, self.batchnorms):
            x = conv(x)
            x = self.activation(x)
            x = batchnorm(x)
        return x


###########################################################
#             Duplication                                 #
#  input   [batch_num, input_channel,    D,   H,   W]     #
#  output  [batch_num, duplication_num,  D,   H,   W]     #
###########################################################
class Duplicate(nn.Module):
    def __init__(self, input_channel, duplication_num, kernel_size):
        super(Duplicate, self).__init__()
        self.duplicate = nn.Conv3d(input_channel, duplication_num, kernel_size=kernel_size, padding=kernel_size//2)
        self.activation = Activation()
        self.batch_norm = nn.BatchNorm3d(duplication_num)
        
    def forward(self, input):
        x = self.duplicate(input)
        x = self.activation(x)
        x = self.batch_norm(x)
        return x


###########################################################
#             Out_layer                                   #
#  input   [batch_num, duplication_num*2,  D,   H,   W]   #
#  output  [batch_num, 2,                  D,   H,   W]   #
###########################################################
class Out_layer(nn.Module):
    def __init__(self, input_channel):
        super(Out_layer, self).__init__()
        self.conv = nn.Conv3d(input_channel, 2, kernel_size=1)
    
    def forward(self, x):
        x = self.conv(x)
        return x


def check_shape(shape, n_layer):
    divider = np.power(2, n_layer)
    D = shape[2]
    H = shape[3]
    W = shape[4]
    if (D % divider != 0):
        raise AssertionError('depth({}) must be a multiple of 2^n_layer({})'.format(D, divider))
    if (H % divider != 0):
        raise AssertionError('height({}) must be a multiple of 2^n_layer({})'.format(D, divider))
    if (W % divider != 0):
        raise AssertionError('width({}) must be a multiple of 2^n_layer({})'.format(D, divider))
# # for testing purpose    
# model = Vnet().cuda()
# x = torch.zeros((1, 1, 128, 128, 128)).cuda()
# y = model(x)
# print(y.shape)