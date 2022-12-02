# encoding: utf-8
"""补充内容见model and loss.ipynb & 自定义双向线性插值滤子（卷积核）.ipynb"""

import numpy as np
import torch
from torchvision import models
from torch import nn
import torch.nn.functional as F


def bilinear_kernel(in_channels, out_channels, kernel_size):
    """Define a bilinear kernel according to in channels and out channels.
    Returns:
        return a bilinear filter tensor
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    bilinear_filter = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float32)
    weight[range(in_channels), range(out_channels), :, :] = bilinear_filter
    return torch.from_numpy(weight)


# pretrained_net = models.vgg16_bn(pretrained=False)


class FCN(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.bb1_1 = BB1(self.input_channel, 4, kernel_size=3, stride=1)
        self.bb1_2 = BB1_2(20, 4, kernel_size=3, stride=1)
        self.bb1_3 = BB1_3(20, 16, kernel_size=3, stride=1, padding=1)
        self.bb1_4 = BB1_4(16, 32, kernel_size=3, stride=1, padding=1)
        self.bb1_5 = BB1_5(32, 32, kernel_size=3, stride=1, padding=1)

        self.upsample_bb2_1 = nn.ConvTranspose2d(16, 16, 4, 2, 1, bias=False)
        self.upsample_bb2_1.weight.data = bilinear_kernel(16, 16, 4)

        self.upsample_bb2_2 = nn.ConvTranspose2d(16, 16, 4, 2, 1, bias=False)
        self.upsample_bb2_2.weight.data = bilinear_kernel(16, 16, 4)

        self.upsample_bb2_3 = nn.ConvTranspose2d(8, 8, 4, 2, 1, bias=False)
        self.upsample_bb2_3.weight.data = bilinear_kernel(8, 8, 4)
        self.conv1 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(8, self.output_channel, kernel_size=3, stride=1, padding=1)
        self.active = nn.ReLU()
        self.bn = nn.BatchNorm2d(16, eps=0.001)

    def forward(self, x):
        x = self.bb1_1(x)
        x = self.bb1_2(x)
        # print(x.shape)
        x = self.bb1_3(x)
        # print(x.shape)
        x = self.bb1_4(x)
        # print(x.shape)
        x = self.bb1_5(x)
        x = self.conv1(x)
        # print(x.shape)
        x = self.upsample_bb2_1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.active(x)
        x = self.upsample_bb2_2(x)
        # print(x.shape)
        x = self.conv3(x)
        x = self.upsample_bb2_3(x)
        # print(x.shape)
        x = self.conv4(x)

        return x


class BB1(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BB1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False, dilation=1, padding=1, **kwargs)
        self.conv2 = nn.Conv2d(in_channels, out_channels, bias=False, dilation=2, padding=2, **kwargs)
        self.conv3 = nn.Conv2d(in_channels, out_channels, bias=False, dilation=3, padding=3, **kwargs)
        self.conv4 = nn.Conv2d(in_channels, out_channels, bias=False, dilation=5, padding=5, **kwargs)
        self.conv5 = nn.Conv2d(in_channels, out_channels, bias=False, dilation=9, padding=9, **kwargs)

        self.bn = nn.BatchNorm2d(20, eps=0.001)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)

        x = torch.cat((x1, x2, x3, x4, x5), 1)

        # x = self.bn(x)
        return F.relu(x, inplace=True)


class BB1_2(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BB1_2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False, dilation=1, padding=1, **kwargs)
        self.conv2 = nn.Conv2d(in_channels, out_channels, bias=False, dilation=2, padding=2, **kwargs)
        self.conv3 = nn.Conv2d(in_channels, out_channels, bias=False, dilation=3, padding=3, **kwargs)
        self.conv4 = nn.Conv2d(in_channels, out_channels, bias=False, dilation=5, padding=5, **kwargs)
        self.conv5 = nn.Conv2d(in_channels, out_channels, bias=False, dilation=9, padding=9, **kwargs)

        self.bn = nn.BatchNorm2d(20, eps=0.001)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.active = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.conv5(x)

        x = torch.cat((x1, x2, x3, x4, x5), 1)
        # x = self.bn(x)
        x = self.active(x)
        x = self.pool(x)

        return x


class BB1_3(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BB1_3, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.active = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)

        # x = self.bn(x)
        x = self.active(x)
        x = self.pool(x)

        return x


class BB1_4(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BB1_4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.active = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)

        # x = self.bn(x)
        x = self.active(x)
        x = self.pool(x)

        return x


class BB1_5(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BB1_5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)

        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.active = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)

        # x = self.bn(x)
        x = self.active(x)

        return x


if __name__ == "__main__":
    import torch as t

    print('-----' * 5)
    rgb = t.randn(1, 1, 300, 300)

    net = FCN()

    out = net(rgb)

    print(out.shape)
