# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------
# python main.py ctdet --arch fatnet --dataset pascal --gpus 0,1 --input_res 384 --num_epochs 210 --lr_step 135,180 --batch_size 32 --exp_id fatnet_pascal_384_daspp_ds4


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from .DCNv2.dcn_v2 import DCN

BN_MOMENTUM = 0.1


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.1, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Pang_unit(nn.Module):  #### basic unit
    def __init__(self, cin, cout, bn):
        super(Pang_unit, self).__init__()
        # if bn==True:
        #     bias = False
        # else:
        #     bias = True
        bias = True
        self.branch0 = BasicConv(cin, cout, kernel_size=3, stride=1, padding=1, bn=bn, bias=bias)
        self.branch1 = BasicConv(cin, cout, kernel_size=1, stride=1, padding=0, bn=bn, bias=bias)
        self.cin = cin
        self.cout = cout

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        if self.cin == self.cout:
            x0 = x0 + x1 + x
        else:
            x0 = x0 + x1
        return x0


class Pang_unit_stride(nn.Module):  #### basic unit
    def __init__(self, cin, cout, bn, dilation):
        super(Pang_unit_stride, self).__init__()
        # if bn==True:
        #     bias = False
        # else:
        #     bias = True
        bias = True

        self.branch0 = BasicConv(cin, cout, kernel_size=3, stride=2, padding=dilation, dilation=dilation, bn=bn,
                                 bias=bias)
        self.branch1 = BasicConv(cin, cout, kernel_size=1, stride=1, padding=0, bn=bn, bias=bias)
        self.cin = cin
        self.cout = cout

    def forward(self, x):
        x0 = self.branch0(x)
        x0 = F.upsample_nearest(x0, scale_factor=2)
        x1 = self.branch1(x)
        if self.cin == self.cout:
            x0 = x1 + x0 + x
        else:
            x0 = x1 + x0
        return x0

class PosePangNet(nn.Module):

    def __init__(self, heads, head_conv, **kwargs):
        # self.inplanes = 64
        # self.deconv_with_bias = False
        self.heads = heads

        super(PosePangNet, self).__init__()

        self.conv1 = BasicConv(3, 16, kernel_size=7, stride=2, padding=3, bias=False, bn=True, relu=True)

        self.features = self._make_layers_pangnet(batch_norm=True)

        self.dcn = nn.Sequential(
            DCN(128, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
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

    def _make_layers_pangnet(self, batch_norm=False):
        layers = nn.ModuleList()
        in_channels = 3
        cfg = [16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 128, 128, 128]
        for idx, v in enumerate(cfg):
            if idx <= 3:
                layers.append(Pang_unit(in_channels, v, bn=batch_norm))
            elif idx > 3 and idx <= 5:
                layers.append(Pang_unit_stride(in_channels, v, bn=batch_norm, dilation=1))
            elif idx > 5 and idx <= 9:
                layers.append(Pang_unit_stride(in_channels, v, bn=batch_norm, dilation=2))
            else:
                layers.append(Pang_unit_stride(in_channels, v, bn=batch_norm, dilation=4))
            in_channels = v
        return layers

    def forward(self, x):

        for layer in self.features:
            x = layer(x)

        x = self.dcn(x)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]

    def init_weights(self, num_layers, pretrained=True):
        # if pretrained:
        print('=> init from pretrianed')
        checkpoint = torch.load('../models/pangnet.pth')
        self.load_state_dict(checkpoint, strict=True)



        print('=> init final conv weights from normal distribution')
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
