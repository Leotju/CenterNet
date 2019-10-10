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
    def __init__(self, in_planes, out_planes, kernel_size, dilation=1, groups=1, relu=True,
                 bn=True, bias=False, tl_size=8):
        super(TLConv, self).__init__()

        out_planes = out_planes
        self.tl_size = tl_size
        # self.conv11 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                         dilation=dilation, groups=groups, bias=bias)
        # self.conv12 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                         dilation=dilation, groups=groups, bias=bias)
        # self.conv21 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                         dilation=dilation, groups=groups, bias=bias)
        # self.conv22 = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
        #                         dilation=dilation, groups=groups, bias=bias)
        self.up = nn.PixelShuffle(upscale_factor=tl_size)
        self.tl_conv_list = nn.ModuleList()
        for h in range(tl_size):
            for w in range(tl_size):
                tl_conv = BasicConv(in_planes, out_planes, kernel_size=kernel_size, stride=tl_size,
                                    padding=1, dilation=dilation, groups=groups, bias=bias,
                                    bn=bn,
                                    relu=relu)
                self.tl_conv_list.append(tl_conv)

    def forward(self, x):
        b, c, h, w = x.size()
        pad = self.tl_size // 2
        w_pad = x.new_zeros(b, c, h, pad)
        h_pad = x.new_zeros(b, c, pad, w + pad * 2)
        x_padding = torch.cat((w_pad, x, w_pad), 3)
        x_padding = torch.cat((h_pad, x_padding, h_pad), 2)
        feats = []
        conv_idx = 0
        for i in range(0, self.tl_size):
            for j in range(0, self.tl_size):
                feat = x_padding[:, :, i:x_padding.size(2) - (self.tl_size - i ),
                       j:x_padding.size(3) - (self.tl_size - j)]
                feat = self.tl_conv_list[conv_idx](feat)
                feats.append(feat.permute(0, 2, 3, 1)[:, :, :, :, None])
                # print(feat.size())
        # for tl_conv in self.tl_conv_list:
        #     feats.append(tl_conv(x).permute(0, 2, 3, 1)[:, :, :, :, None])
        outs = torch.cat(feats, 4)
        outs = outs.view(outs.size(0), outs.size(1), outs.size(2), -1).view(outs.size(0), outs.size(1),
                                                                            outs.size(2), -1).permute(0, 3, 1, 2)
        outs = self.up(outs)
        return outs
