import torch.nn as nn
import torch


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# class TLConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(TLConv, self).__init__()
#         stride = stride * 2
#         self.conv11 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.conv12 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.conv21 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.conv22 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.up = nn.PixelShuffle(upscale_factor=2)
#
#     def forward(self, x):
#         x11 = self.conv11(x)
#         x12 = self.conv12(x)
#         x21 = self.conv21(x)
#         x22 = self.conv22(x)
#         # B C H W
#         x11v = x11.permute(0, 2, 3, 1)[:, :, :, :, None]
#         x12v = x12.permute(0, 2, 3, 1)[:, :, :, :, None]
#         x21v = x21.permute(0, 2, 3, 1)[:, :, :, :, None]
#         x22v = x22.permute(0, 2, 3, 1)[:, :, :, :, None]
#
#
#         feats = torch.cat((x11v, x12v, x21v, x22v), 4)
#         feats = feats.view(feats.size(0), feats.size(1), feats.size(2), -1).view(feats.size(0), feats.size(1), feats.size(2), -1).permute(0, 3, 1, 2)
#
#
#         out = self.up(feats)
#         return out


# class TLConvUnit(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(TLConvUnit, self).__init__()
#         stride = stride * 2
#         out_planes = out_planes
#         self.conv11 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                                 dilation=dilation, groups=groups, bias=bias)
#         self.conv12 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                                 dilation=dilation, groups=groups, bias=bias)
#         self.conv21 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                                 dilation=dilation, groups=groups, bias=bias)
#         self.conv22 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                                 dilation=dilation, groups=groups, bias=bias)
#         self.up = nn.PixelShuffle(upscale_factor=2)
#
#     def forward(self, x):
#         x11 = self.conv11(x)
#         x12 = self.conv12(x)
#         x21 = self.conv21(x)
#         x22 = self.conv22(x)
#         # B C H W
#         x11v = x11.permute(0, 2, 3, 1)[:, :, :, :, None]
#         x12v = x12.permute(0, 2, 3, 1)[:, :, :, :, None]
#         x21v = x21.permute(0, 2, 3, 1)[:, :, :, :, None]
#         x22v = x22.permute(0, 2, 3, 1)[:, :, :, :, None]
#
#         feats = torch.cat((x11v, x12v, x21v, x22v), 4)
#         feats = feats.view(feats.size(0), feats.size(1), feats.size(2), -1).view(feats.size(0), feats.size(1),
#                                                                                  feats.size(2), -1).permute(0, 3, 1, 2)
#
#         return feats
#
#
# class TLConv(nn.Module):
#     def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
#                  bn=True, bias=False):
#         super(TLConv, self).__init__()
#         out_planes = out_planes // 2
#         self.conv11 = TLConvUnit(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                                 dilation=dilation, groups=groups, bias=bias)
#         self.conv12 = TLConvUnit(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#                                 dilation=dilation, groups=groups, bias=bias)
#         # self.conv21 = TLConvUnit(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#         #                         dilation=dilation, groups=groups, bias=bias)
#         # self.conv22 = TLConvUnit(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
#         #                         dilation=dilation, groups=groups, bias=bias)
#         self.up = nn.PixelShuffle(upscale_factor=2)
#
#     def forward(self, x):
#         x11 = self.conv11(x)
#         x12 = self.conv12(x)
#         # x21 = self.conv21(x)
#         # x22 = self.conv22(x)
#         # B C H W
#         x11v = x11.permute(0, 2, 3, 1)[:, :, :, :, None]
#         x12v = x12.permute(0, 2, 3, 1)[:, :, :, :, None]
#         # x21v = x21.permute(0, 2, 3, 1)[:, :, :, :, None]
#         # x22v = x22.permute(0, 2, 3, 1)[:, :, :, :, None]
#
#         # feats = torch.cat((x11v, x12v, x21v, x22v), 4)
#         feats = torch.cat((x11v, x12v), 4)
#         feats = feats.view(feats.size(0), feats.size(1), feats.size(2), -1).view(feats.size(0), feats.size(1),
#                                                                                  feats.size(2), -1).permute(0, 3, 1, 2)
#         feats = self.up(feats)
#         return feats


class TLConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(TLConv, self).__init__()
        # stride = stride * 2
        out_planes = out_planes // 4

        self.conv11 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)
        self.conv12 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)
        self.conv21 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation, groups=groups, bias=bias)
        self.conv22 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias)
        self.up = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        x11 = self.conv11(x)

        x_pad1 = torch.cat((x[:, :, 1:, :], x.new_zeros((x.size(0), x.size(1), 1, x.size(3)))), 2)
        x12 = self.conv12(x_pad1)

        x_pad2 = torch.cat((x[:, :, :, 1:], x.new_zeros((x.size(0), x.size(1), x.size(2), 1))), 3)
        x21 = self.conv21(x_pad2)

        x_pad3 = torch.cat(torch.cat((x[:, :, 1:, 1:], x.new_zeros((x.size(0), x.size(1), x.size(2) - 1, 1))), 3), x.new_zeros((x.size(0), x.size(1), 1, x.size(3))), 2)
        x22 = self.conv22(x_pad3)

        feats = torch.cat((x11, x12, x21, x22), 1)

        # feats = self.conv2(feats)

        return feats
