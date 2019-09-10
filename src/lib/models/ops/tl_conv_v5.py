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

class TLConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False, tile_size=2):
        super(TLConv, self).__init__()
        # stride = stride * 2
        tile_chn = max(in_planes // (tile_size * tile_size), 1)
        self.trans_conv = BasicConv(tile_chn * tile_size * tile_size, out_planes, 1, 1, 0)

        # self.conv11 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                         dilation=dilation, groups=groups, bias=bias)
        # self.conv12 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                         dilation=dilation, groups=groups, bias=bias)
        # self.conv21 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                        dilation=dilation, groups=groups, bias=bias)
        # self.conv22 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                         dilation=dilation, groups=groups, bias=bias)
        #
        self.tile_size = tile_size
        self.conv_list = nn.ModuleList()
        for i in range(tile_size * tile_size):
            self.conv_list.append(BasicConv(in_planes, tile_chn, kernel_size=kernel_size, stride=stride, padding=padding,
                                dilation=dilation, groups=groups, bias=bias))

    def forward(self, x):
        # x11 = self.conv11(x)

        # x_pad1 = torch.cat((x[:, :, 1:, :], x.new_zeros((x.size(0), x.size(1), 1, x.size(3)))), 2)
        # x12 = self.conv12(x_pad1)
        #
        # x_pad2 = torch.cat((x[:, :, :, 1:], x.new_zeros((x.size(0), x.size(1), x.size(2), 1))), 3)
        # x21 = self.conv21(x_pad2)
        #
        # x_pad3 = torch.cat((torch.cat((x[:, :, 1:, 1:], x.new_zeros((x.size(0), x.size(1), x.size(2) - 1, 1))), 3), x.new_zeros((x.size(0), x.size(1), 1, x.size(3)))), 2)
        # x22 = self.conv22(x_pad3)

        output = []
        output.append(self.conv_list[0](x))
        conv_idx = 0
        for i in range(1, self.tile_size):
            conv_idx += 1
            x_pad = torch.cat((x[:, :, i:, :], x.new_zeros((x.size(0), x.size(1), i, x.size(3)))), 2)
            output.append(self.conv_list[conv_idx](x_pad))

        for i in range(1, self.tile_size):
            conv_idx += 1
            x_pad = torch.cat((x[:, :, :, i:], x.new_zeros((x.size(0), x.size(1), x.size(2), i))), 3)
            output.append(self.conv_list[conv_idx](x_pad))

        for j in range(1, self.tile_size):
            for k in range(1, self.tile_size):
                conv_idx += 1
                x_pad = torch.cat((torch.cat((x[:, :, j:, k:], x.new_zeros((x.size(0), x.size(1), x.size(2) - j, k))),
                                              3), x.new_zeros((x.size(0), x.size(1), j, x.size(3)))), 2)
                output.append(self.conv_list[conv_idx](x_pad))

        feats = torch.cat(output, 1)
        feats = self.trans_conv(feats)
        # feats = self.conv2(feats)

        return feats
