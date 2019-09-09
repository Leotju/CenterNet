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
# from ..ops.tl_conv import TLConv
from ..ops.tl_conv_v4 import TLConv
from ..ops.basic_conv import BasicConv

BN_MOMENTUM = 0.1

class MKConv(nn.Module):
    def __init__(self, in_channels, kernel_list = [1, 3, 5, 9]):
        super(MKConv, self).__init__()
        self.multi_kenel_convs = nn.ModuleList()
        for kernel in kernel_list:
            self.multi_kenel_convs.append(BasicConv(in_channels, in_channels, kernel_size=kernel, stride=1, padding=kernel // 2))

    def forward(self, x):
        outs = []
        for mkconv in self.multi_kenel_convs:
            outs.append(mkconv(x))
        feats = torch.cat(outs, 1)
        return feats





class PosePangNet(nn.Module):

    def __init__(self, heads, head_conv, **kwargs):
        # self.inplanes = 64
        # self.deconv_with_bias = False
        self.heads = heads

        super(PosePangNet, self).__init__()

        self.conv1_1 = BasicConv(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = BasicConv(64, 64, kernel_size=3, stride=1, padding=1)

        # self.conv2_1 = BasicConv(64, 32, kernel_size=3, stride=1, padding=2, dilation=2)
        # self.conv2_2 = BasicConv(32, 32, kernel_size=3, stride=1, padding=2, dilation=2)
        #
        # self.conv3_1 = BasicConv(32, 16, kernel_size=3, stride=1, padding=4, dilation=4)
        # self.conv3_2 = BasicConv(16, 16, kernel_size=3, stride=1, padding=4, dilation=4)
        # self.conv3_3 = BasicConv(16, 16, kernel_size=3, stride=1, padding=4, dilation=4)
        #
        # self.conv4_1 = BasicConv(16, 8, kernel_size=3, stride=1, padding=8, dilation=8)
        # self.conv4_2 = BasicConv(8, 8, kernel_size=3, stride=1, padding=8, dilation=8)
        # self.conv4_3 = BasicConv(8, 8, kernel_size=3, stride=1, padding=8, dilation=8)
        #
        # self.conv5_1 = BasicConv(8, 8, kernel_size=3, stride=1, padding=16, dilation=16)
        # self.conv5_2 = BasicConv(8, 8, kernel_size=3, stride=1, padding=16, dilation=16)
        # self.conv5_3 = BasicConv(8, 8, kernel_size=3, stride=1, padding=16, dilation=16)



        self.conv2_1 = TLConv(64, 32, kernel_size=3, stride=1, padding=2, dilation=2, tile_size=3)
        self.conv2_2 = TLConv(32, 32, kernel_size=3, stride=1, padding=2, dilation=2, tile_size=3)

        self.conv3_1 = TLConv(32, 16, kernel_size=3, stride=1, padding=4, dilation=4, tile_size=3)
        self.conv3_2 = TLConv(16, 16, kernel_size=3, stride=1, padding=4, dilation=4, tile_size=3)
        self.conv3_3 = TLConv(16, 16, kernel_size=3, stride=1, padding=4, dilation=4, tile_size=3)

        self.conv4_1 = TLConv(16, 8, kernel_size=3, stride=1, padding=8, dilation=8, tile_size=3)
        self.conv4_2 = TLConv(8, 8, kernel_size=3, stride=1, padding=8, dilation=8, tile_size=3)
        self.conv4_3 = TLConv(8, 8, kernel_size=3, stride=1, padding=8, dilation=8, tile_size=3)

        self.conv5_1 = TLConv(8, 8, kernel_size=3, stride=1, padding=16, dilation=16, tile_size=3)
        self.conv5_2 = TLConv(8, 8, kernel_size=3, stride=1, padding=16, dilation=16, tile_size=3)
        self.conv5_3 = TLConv(8, 8, kernel_size=3, stride=1, padding=16, dilation=16, tile_size=3)

        # self.conv2_1 = TLConv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1, tile_size=7)
        # self.conv2_2 = TLConv(32, 32, kernel_size=3, stride=1, padding=1, dilation=1, tile_size=7)
        #
        # self.conv3_1 = TLConv(32, 16, kernel_size=3, stride=1, padding=1, dilation=1, tile_size=7)
        # self.conv3_2 = TLConv(16, 16, kernel_size=3, stride=1, padding=1, dilation=1, tile_size=7)
        # self.conv3_3 = TLConv(16, 16, kernel_size=3, stride=1, padding=1, dilation=1, tile_size=7)
        #
        # self.conv4_1 = TLConv(16, 8, kernel_size=3, stride=1, padding=1, dilation=1, tile_size=3)
        # self.conv4_2 = TLConv(8, 8, kernel_size=3, stride=1, padding=1, dilation=1, tile_size=3)
        # self.conv4_3 = TLConv(8, 8, kernel_size=3, stride=1, padding=1, dilation=1, tile_size=3)
        #
        # self.conv5_1 = TLConv(8, 8, kernel_size=3, stride=1, padding=1, dilation=1, tile_size=3)
        # self.conv5_2 = TLConv(8, 8, kernel_size=3, stride=1, padding=1, dilation=1, tile_size=3)
        # self.conv5_3 = TLConv(8, 8, kernel_size=3, stride=1, padding=1, dilation=1, tile_size=3)


        self.frn = nn.Sequential(
            self.conv1_1, self.conv1_2, self.conv2_1, self.conv2_2, self.conv3_1, self.conv3_2, self.conv3_3,
            self.conv4_1, self.conv4_2, self.conv4_3, self.conv5_1, self.conv5_2, self.conv5_3,
        )


        self.multi_kernel_pred = MKConv(in_channels=8, kernel_list=[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25])


        # self.dcn = nn.Sequential(
        #     DCN(128, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
        #     nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     BasicConv(64, 64, kernel_size=3, stride=1, padding=1),
        #     DCN(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
        #     nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     BasicConv(64, 64, kernel_size=3, stride=1, padding=1),
        #     DCN(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
        #     nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     BasicConv(64, 64, kernel_size=3, stride=1, padding=1),
        # )

        # self.dcn = nn.Sequential(
        #     DCN(512, 256, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
        #     nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     BasicConv(256, 256, kernel_size=3, stride=1, padding=1),
        #     DCN(256, 128, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
        #     nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     BasicConv(128, 64, kernel_size=3, stride=1, padding=1),
        #     DCN(64, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, deformable_groups=1),
        #     nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
        #     nn.UpsamplingBilinear2d(scale_factor=2),
        #     BasicConv(64, 64, kernel_size=3, stride=1, padding=1),
        # )

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
                    BasicConv(96, head_conv, kernel_size=3, padding=1, bias=True, bn=True, relu=True),
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

    def forward(self, x):

        x = self.frn(x)
        x = self.multi_kernel_pred(x)
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
