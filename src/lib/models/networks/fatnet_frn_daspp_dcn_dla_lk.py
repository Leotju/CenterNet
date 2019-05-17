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


class dense_aspp(nn.Module):
    def __init__(self):
        super(dense_aspp, self).__init__()
        bias = True
        bn = True
        # self.conv1  =
        self.conv_d3 = BasicConv(128, 24, kernel_size=3, stride=1, padding=3, dilation=3, bn=bn, bias=bias)
        self.conv_d6 = BasicConv(128 + 24, 24, kernel_size=3, stride=1, padding=6, dilation=6, bn=bn, bias=bias)
        self.conv_d12 = BasicConv(128 + 48, 24, kernel_size=3, stride=1, padding=12, dilation=12, bn=bn, bias=bias)
        self.conv_d18 = BasicConv(128 + 72, 24, kernel_size=3, stride=1, padding=18, dilation=18, bn=bn, bias=bias)
        self.conv_d24 = BasicConv(128 + 96, 24, kernel_size=3, stride=1, padding=24, dilation=24, bn=bn, bias=bias)

        self.trans = BasicConv(128 + 120, 128, kernel_size=1, stride=1, padding=0, bn=bn, bias=bias)

    def forward(self, x):
        d3 = self.conv_d3(x)
        d6 = self.conv_d6(torch.cat((x, d3), 1))
        # d9 = self.conv_d9(torch.cat((x, d3, d6), 1))
        d12 = self.conv_d12(torch.cat((x, d3, d6), 1))
        d18 = self.conv_d18(torch.cat((x, d3, d6, d12), 1))
        d24 = self.conv_d24(torch.cat((x, d3, d6, d12, d18), 1))
        out = self.trans(torch.cat((x, d3, d6, d12, d18, d24), 1))

        return out


class PosePangNet(nn.Module):

    def __init__(self, heads, head_conv, **kwargs):
        # self.inplanes = 64
        # self.deconv_with_bias = False
        self.heads = heads

        super(PosePangNet, self).__init__()

        self.conv1 = BasicConv(3, 16, kernel_size=7, stride=1, padding=3, bias=False, bn=True, relu=True)

        self.features = self._make_layers_pangnet(batch_norm=True)
        self.dense_aspp = dense_aspp()

        self.dcn = nn.Sequential(
            DCN(128, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            BasicConv(64, 64, kernel_size=3, stride=1, padding=1),
            DCN(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            BasicConv(64, 64, kernel_size=3, stride=1, padding=1),
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
                    BasicConv(64, head_conv, kernel_size=9, padding=4, bias=True, bn=True, relu=True),
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
        for ic, v in enumerate(cfg):
            v = v * 1
            if ic <= 1:
                layers.append(Pang_unit(in_channels, v, bn=batch_norm))
            elif ic > 1 and ic <= 5:
                layers.append(Pang_unit_stride(in_channels, v, bn=batch_norm, dilation=1))
            elif ic > 5 and ic <= 9:
                layers.append(Pang_unit_stride(in_channels, v, bn=batch_norm, dilation=2))
            else:
                layers.append(Pang_unit_stride(in_channels, v, bn=batch_norm, dilation=4))
            in_channels = v
        return layers

    def forward(self, x):

        x = self.conv1(x)
        # x = F.max_pool2d(x, kernel_size=2, stride=2)
        # x = F.avg_pool2d(x, kernel_size=4, stride=4)

        # for layer in self.features:
        #     id += 1
        #     if id == 4 or id == 8:
        #         x = F.max_pool2d(x, kernel_size=2, stride=2)
        #         x = layer(x)
        #     else:
        #         x = layer(x)

        for layer in self.features:
            x = layer(x)

        # x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = self.dense_aspp(x)
        x = self.dcn(x)

        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # x = self.maxpool(x)

        # x = self.layer1(x)
        # x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        #
        # x = self.deconv_layers(x)
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
