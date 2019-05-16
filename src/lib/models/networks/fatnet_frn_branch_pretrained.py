'''
Modified from https://github.com/pytorch/vision.git
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .DCNv2.dcn_v2 import DCN

### chose P2 for 76.3
cfg = {
    'P1': [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512],
    'P3': [64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64, 64],
    'P2': [16, 16, 32, 32, 32, 32, 64, 64, 64, 64, 64, 128, 128],
}

BN_MOMENTUM = 0.1

# def pangnet_branch(num_classes):
#     # return PANGNet(make_layers_pangnet(cfg['P2'], batch_norm=True))
#     return PANGNet(make_layers_pangnet(cfg['P2'], batch_norm=True))

def get_pose_net(num_layers, heads, head_conv):
    model = PosePangNet(heads, head_conv=head_conv)
    model.init_weights(num_layers, pretrained=True)
    return model

class PosePangNet(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, heads, head_conv):
        super(PosePangNet, self).__init__()
        self.heads = heads
        features = self._make_layers_pangnet(cfg['P2'], batch_norm=True)

        self.features = features[0]
        self.features2 = features[1]

        self.glp = nn.AdaptiveAvgPool2d(output_size=1)

        self.dsapp = dense_aspp()
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



    def forward(self, x):
        x1 = x
        id = 0
        for layer, layer2 in zip(self.features, self.features2):
            if id == 0:
                x1 = layer2(x1)
                x = layer(x)
            else:
                x2 = layer2(x1)
                x1 = x1 + x2
                x = layer(torch.cat((x, x1), 1))
            id += 1
        print(x.max())
        x = self.dsapp(x)
        x = self.dcn(x)

        return x

    def init_weights(self, num_layers, pretrained=True):
        # if pretrained:
        # print('=> init resnet deconv weights from normal distribution')
        pretrained_state_dict = torch.load('../models/pangnet_branch.pth')
        print('=> loading pretrained model {}'.format('pangnet_branch'))
        self.load_state_dict(pretrained_state_dict, strict=False)

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

    def _make_layers_pangnet(self, cfg, batch_norm=False):
        layers = nn.ModuleList()
        layers2 = nn.ModuleList()
        in_channels = 3
        for idx, v in enumerate(cfg):
            if idx <= 1:
                layers.append(Pang_unit(in_channels, v, bn=batch_norm))
            else:
                layers.append(Pang_unit_stride(in_channels, v, bn=batch_norm))
            # if idx == 0:
            #     in_channels = v
            # else:
            in_channels = v + 16
            if idx == 0:
                layers2.append(Pang_unit(3, 16, bn=batch_norm))
            else:
                layers2.append(Pang_unit(16, 16, bn=batch_norm))
        return layers, layers2






class Pang_unit(nn.Module):
    def __init__(self, cin, cout, bn):
        super(Pang_unit, self).__init__()
        bias = True
        self.branch0 = BasicConv(cin, cout, kernel_size=3, stride=1, padding=1, bn=bn, bias=bias)
        # self.branch1 = BasicConv(cin, cout, kernel_size=1, stride=1, padding=0, bn=bn, bias=bias)

    def forward(self, x):
        x0 = self.branch0(x)
        # x1 = self.branch1(x)
        # x0 = x0 + x1
        return x0

class Pang_unit_stride(nn.Module):  #### basic unit
    def __init__(self, cin, cout, bn):
        super(Pang_unit_stride, self).__init__()
        bias = True
        self.branch0 = BasicConv(cin, cout, kernel_size=3, stride=2, padding=1, bn=bn, bias=bias)
        self.branch1 = BasicConv(cin, cout, kernel_size=1, stride=1, padding=0, bn=bn, bias=bias)

    def forward(self, x):
        x0 = self.branch0(x)
        x0 = F.upsample_nearest(x0, scale_factor=2)
        x1 = self.branch1(x)
        x0 = x0 + x1
        return x0

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

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