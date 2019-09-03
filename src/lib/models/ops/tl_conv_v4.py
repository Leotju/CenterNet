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
                 bn=True, bias=False, tile_size=1):
        super(TLConv, self).__init__()
        # stride = stride * 2
        self.tile_size = tile_size
        out_planes_div = out_planes // (self.tile_size * self.tile_size)
        self.convs = nn.ModuleList()
        # print(out_planes_div * self.tile_size * self.tile_size)
        self.conv_trans = BasicConv(out_planes_div * self.tile_size * self.tile_size, out_planes, kernel_size=1, stride=1, padding=0)
        for i in range(tile_size * tile_size):
            self.convs.append(BasicConv(in_planes, out_planes_div, kernel_size=kernel_size, stride=stride, padding=padding,
                                 dilation=dilation, groups=groups, bias=bias))

    def forward(self, x):
        b, c, h, w = x.size()
        pad = self.tile_size // 2
        w_pad = x.new_zeros(b, c, h, pad)
        h_pad = x.new_zeros(b, c, pad, w + pad * 2)
        x_padding = torch.cat((w_pad, x, w_pad), 3)
        x_padding = torch.cat((h_pad, x_padding, h_pad), 2)
        outs = []
        conv_idx = 0
        for i in range(0, self.tile_size):
            for j in range(0, self.tile_size):
                feat = x_padding[:, :, i:x_padding.size(2) - (self.tile_size - i - 1),
                       j:x_padding.size(3) - (self.tile_size - j - 1)]
                feat = self.convs[conv_idx](feat)
                outs.append(feat)
        out_feat = torch.cat(outs, 1)
        out_feat = self.conv_trans(out_feat)
        return out_feat
