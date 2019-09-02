# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from .DCNv2.dcn_v2 import DCN
from ..ops.tl_conv import TLConv
from ..ops.basic_conv import BasicConv

BN_MOMENTUM = 0.1


class Pang_unit(nn.Module):  #### basic unit
    def __init__(self, cin, cout, bn, dilation=1):
        super(Pang_unit, self).__init__()
        # if bn==True:
        #     bias = False
        # else:
        #     bias = True
        bias = True
        self.branch0 = TLConv(cin, cout, kernel_size=3, stride=1, padding=1, bn=bn, bias=bias, dilation=dilation)
        self.branch1 = BasicConv(cin, cout, kernel_size=1, stride=1, padding=0, bn=bn, bias=bias)
        self.cin = cin
        self.cout = cout

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x0 = x0 + x1
        return x0


class Pang_unit_stride(nn.Module):  #### basic unit
    def __init__(self, cin, cout, bn, dilation):
        super(Pang_unit_stride, self).__init__()
        bias = False

        self.branch0 = TLConv(cin, cout, kernel_size=3, stride=2, padding=dilation, dilation=dilation, bn=bn,
                              bias=bias)
        self.branch1 = BasicConv(cin, cout, kernel_size=1, stride=1, padding=0, bn=bn, bias=bias)
        self.cin = cin
        self.cout = cout

    def forward(self, x):
        x0 = self.branch0(x)
        x0 = F.upsample_nearest(x0, scale_factor=2)
        x1 = self.branch1(x)

        x0 = x1 + x0
        return x0


class Multi_Pooling(nn.Module):  #### basic unit
    def __init__(self):
        super(Multi_Pooling, self).__init__()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.pool4 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.pool8 = nn.AvgPool2d(kernel_size=8, stride=8)
        self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)

        self.pool16 = nn.AvgPool2d(kernel_size=16, stride=16)
        self.up16 = nn.UpsamplingBilinear2d(scale_factor=16)

    def forward(self, x):
        p2 = self.up2(self.pool2(x))
        p4 = self.up4(self.pool4(x))
        p8 = self.up8(self.pool8(x))
        p16 = self.up16(self.pool16(x))
        gap = F.upsample_bilinear(F.adaptive_avg_pool2d(x, output_size=1), size=(x.size(2), x.size(3)))
        out = x + p2 + p4 + p8 + p16 + gap
        return out


class PosePangNet(nn.Module):

    def __init__(self, heads, head_conv, **kwargs):
        # self.inplanes = 64
        # self.deconv_with_bias = False
        self.heads = heads

        super(PosePangNet, self).__init__()

        self.conv1 = BasicConv(3, 16, kernel_size=7, stride=2, padding=3, bias=False, bn=True, relu=True)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.features = self._make_layers_pangnet(batch_norm=True)

        self.trans_conv = BasicConv(128 + 64 + 32, 128, kernel_size=3, stride=1, padding=1, bias=False, bn=True,
                                    relu=True)

        self.mlp = Multi_Pooling()

        self.dcn = nn.Sequential(
            DCN(128, 128, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            BasicConv(128, 128, kernel_size=3, stride=1, padding=1),
            DCN(128, 128, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
            BasicConv(128, 64, kernel_size=3, stride=1, padding=1),
            DCN(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            BasicConv(64, 64, kernel_size=3, stride=1, padding=1),
        )

        for head in sorted(self.heads):
            num_output = self.heads[head]

            if head_conv > 0:
                # fc = nn.Sequential(
                #     nn.Conv2d(128, head_conv,
                #               kernel_size=3, padding=1, bias=True),
                #     nn.ReLU(inplace=True),
                #     nn.Conv2d(head_conv, num_output,
                #               kernel_size=1, stride=1, padding=0))

                fc = nn.Sequential(
                    BasicConv(64, head_conv, kernel_size=3, padding=1, bias=True, bn=True, relu=True),
                    nn.Conv2d(head_conv, num_output, kernel_size=1, stride=1, padding=0))
                # BasicConv(head_conv, num_output, kernel_size=1, padding=0, bias=True, bn=True, relu=False))


            else:
                fc = nn.Conv2d(
                    in_channels=128,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            self.__setattr__(head, fc)

        # self.final_layer = nn.ModuleList(self.final_layer)

    def _make_layers_pangnet(self, batch_norm=True):
        layers = nn.ModuleList()
        in_channels = 16
        cfg = [16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128]
        dilation = [1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8]
        for ic, v in enumerate(cfg):
            v = v * 1
            if ic <= 1:
                layers.append(Pang_unit(in_channels, v, dilation=dilation[ic], bn=batch_norm))
            else:
                layers.append(Pang_unit_stride(in_channels, v, bn=batch_norm, dilation=dilation[ic]))
            in_channels = v
        return layers

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        index = [5, 9, 12]
        mlp_ind = [1, 5, 9]
        output = []

        for id, layer in enumerate(self.features):
            x = layer(x)
            if id in index:
                output.append(x)
            if id in mlp_ind:
                x = self.mlp(x)

        x = torch.cat(output, 1)
        x = self.trans_conv(x)
        x = self.dcn(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self, num_layers, pretrained=True):
        # if pretrained:
        # print('=> init resnet deconv weights from normal distribution')

        # print('=> init final conv weights from normal distribution')
        for head in self.heads:
            final_layer = self.__getattr__(head)
            for i, m in enumerate(final_layer.modules()):
                if isinstance(m, nn.Conv2d):
                    # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                    # print('=> init {}.bias as 0'.format(name))
                    if m.weight.shape[0] == self.heads[head]:
                        if 'hm' in head:
                            nn.init.constant_(m.bias, -2.19)
                        else:
                            nn.init.normal_(m.weight, std=0.001)
                            nn.init.constant_(m.bias, 0)
            # pretrained_state_dict = torch.load(pretrained)

            # url = model_urls['resnet{}'.format(num_layers)]
            # pretrained_state_dict = model_zoo.load_url(url)
            # print('=> loading pretrained model {}'.format(url))
            # self.load_state_dict(pretrained_state_dict, strict=False)
        # else:
        #     print('=> imagenet pretrained model dose not exist')
        #     print('=> please download it first')
        #     raise ValueError('imagenet pretrained model does not exist')


def get_pose_net(num_layers, heads, head_conv):
    model = PosePangNet(heads, head_conv=head_conv)
    model.init_weights(num_layers, pretrained=True)
    return model
