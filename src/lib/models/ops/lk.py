import torch
import torch.nn as nn
from .basic_conv import BasicConv


# class BasicConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(BasicConv, self).__init__()
#         self.out_channels = out_planes
#         self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                               dilation=dilation, groups=groups, bias=bias)
#         self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
#         self.relu = nn.ReLU() if relu else None
#
#     def forward(self, x):
#         x = self.conv(x)
#         if self.bn is not None:
#             x = self.bn(x)
#         if self.relu is not None:
#             x = self.relu(x)
#         return x

class LkConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super(LkConv, self).__init__()
        self.conv1 = BasicConv(in_planes, in_planes // 2, 3, 1, 1)
        self.conv2 = BasicConv(in_planes, out_planes, 3, 1, 1)
        self.lk_v_conv = BasicConv(in_planes // 2, in_planes // 2, kernel_size=(kernel_size, 1),
                                   padding=(kernel_size // 2, 0))
        self.lk_h_conv = BasicConv(in_planes // 2, in_planes // 2, kernel_size=(1, kernel_size),
                                   padding=(0, kernel_size // 2))

    def forward(self, x):
        x = self.conv1(x)
        h_x = self.lk_h_conv(x)
        v_x = self.lk_v_conv(x)
        x = torch.cat((h_x, v_x), 1)
        x = self.conv2(x)
        return x
