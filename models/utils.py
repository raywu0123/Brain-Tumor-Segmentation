import torch
from torch import nn
import numpy as np


def normalize_batch_image(batch_image):
    assert(batch_image.ndim == 4)
    std = np.std(batch_image, axis=(1, 2, 3), keepdims=True)
    std_is_zero = std == 0
    batch_image = (batch_image - np.mean(batch_image, axis=(1, 2, 3), keepdims=True)) \
        / (std + std_is_zero.astype(float))
    return batch_image


def co_shuffle(batch_data, batch_label):
    assert(len(batch_data) == len(batch_label))
    p = np.random.permutation(len(batch_data))
    batch_data = batch_data[p]
    batch_label = batch_label[p]
    return batch_data, batch_label


def get_tensor_from_array(array):
    tensor = torch.Tensor(array)
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return tensor


def summarize_logs(logs: [dict]) -> dict:
    summary = {}
    if len(logs) == 0:
        return summary

    for key in logs[0].keys():
        summary[key] = np.mean([d[key] for d in logs])

    summary['data_count'] = len(logs)
    return summary


class SelfAttention3D(nn.Module):

    def __init__(self, in_dim):
        super(SelfAttention3D, self).__init__()
        self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # N = D x H x W

        batchsize, C, D, H, W = x.size()
        N = D * H * W
        assert(C >= 8)

        proj_query = self.query_conv(x).view(batchsize, -1, N).permute(0, 2, 1)  # B X N x C'
        proj_key = self.key_conv(x).view(batchsize, -1, N)  # B X C' x N
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)  # B X N X N
        proj_value = self.value_conv(x).view(batchsize, C, N)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(x.size())

        out = self.gamma * out + x
        return out


class MobileSelfAttention3D(nn.Module):

    def __init__(self, in_dim):
        super(MobileSelfAttention3D, self).__init__()

        self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.d_query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.h_query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.w_query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batchsize, C, D, H, W = x.size()
        N = D * H * W
        assert(C >= 8)

        key = self.key_conv(x).view(batchsize, N, -1) # N x C'

        d_query = self.d_query_conv(x)
        d_query = torch.mean(torch.mean(d_query, dim=4), dim=3).view(batchsize, -1, D) # C' * D
        d_energy = torch.bmm(key, d_query).view(batchsize, D, H, W, D)
        d_attention = nn.Softmax(dim=-1)(d_energy)

        h_query = self.h_query_conv(x)
        h_query = torch.mean(torch.mean(h_query, dim=4), dim=2).view(batchsize, -1, H)
        h_energy = torch.bmm(key, h_query).view(batchsize, D, H, W, H)
        h_attention = nn.Softmax(dim=-1)(h_energy)

        w_query = self.h_query_conv(x)
        w_query = torch.mean(torch.mean(w_query, dim=3), dim=2).view(batchsize, -1, W)
        w_energy = torch.bmm(key, w_query).view(batchsize, D, H, W, W)
        w_attention = nn.Softmax(dim=-1)(w_energy)

        value = self.value_conv(x)
        out = torch.einsum('bclmn,bijkl,bijkm,bijkn->bcijk', (value, d_attention, h_attention, w_attention))
        out = self.gamma * out + x
        return out