from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
import os

from .networks.msra_resnet import get_pose_net
from .networks.dlav0 import get_pose_net as get_dlav0
from .networks.pose_dla_dcn import get_pose_net as get_dla_dcn
from .networks.resnet_dcn import get_pose_net as get_pose_net_dcn
from .networks.large_hourglass import get_large_hourglass_net
from .networks.vgg_dcn import get_pose_net as get_pose_net_vgg_dcn
from .networks.fatnet import get_pose_net as get_pose_net_fatnet

from .networks.fatnet_frn_dla import get_pose_net as get_pose_net_fatnet_frn_dla_daspp
from .networks.fatnet_frn_mb_daspp_dcn import get_pose_net as get_pose_net_fatnet_frn_mb_daspp_dcn
from .networks.fatnet_frn_daspp_dcn_dla import get_pose_net as get_pose_net_fatnet_frn_daspp_dcn_dla
from .networks.fatnet_frn_daspp_dcn_dla_att import get_pose_net as get_pose_net_fatnet_frn_daspp_dcn_dla_att
from .networks.fatnet_frn_branch_pretrained import get_pose_net as get_pose_net_fatnet_frn_branch_daspp_dcn_pretrained
from .networks.fatnet_frn_daspp_dcn_dla_lk import get_pose_net as get_pose_net_fatnet_frn_daspp_dcn_dla_lk

from .networks.fatnet_frn import get_pose_net as get_pose_net_fatnet_frn
from .networks.fatnet_daspp_dcn import get_pose_net as get_pose_net_fatnet_daspp_dcn
from .networks.fatnet_daspp_dcn_dla import get_pose_net as get_pose_net_fatnet_daspp_dcn_dla
from .networks.fatnet_daspp_dcn_dla_lk import get_pose_net as get_pose_net_fatnet_daspp_dcn_dla_lk
from .networks.fatnet_daspp_dcn_dla_lk_416 import get_pose_net as get_pose_net_fatnet_daspp_dcn_dla_lk_416
from .networks.fatnet_daspp_dcn_dla_lk_dr import get_pose_net as get_pose_net_fatnet_daspp_dcn_dla_lk_dr
from .networks.fatnet_daspp_dcn_dla_lk_se import get_pose_net as get_pose_net_fatnet_daspp_dcn_dla_lk_se

_model_factory = {
    'res': get_pose_net,  # default Resnet with deconv
    'dlav0': get_dlav0,  # default DLAup
    'dla': get_dla_dcn,
    'resdcn': get_pose_net_dcn,
    'hourglass': get_large_hourglass_net,
    'vgg': get_pose_net_vgg_dcn,
    'fatnet': get_pose_net_fatnet,
    'fatnetdasppdcn': get_pose_net_fatnet_daspp_dcn,
    'fatnetdasppdcndla': get_pose_net_fatnet_daspp_dcn_dla,
    'fatnetdasppdcndlalk': get_pose_net_fatnet_daspp_dcn_dla_lk,
    'fatnetdasppdcndlalk416': get_pose_net_fatnet_daspp_dcn_dla_lk_416,
    'fatnetdasppdcndlalkdr': get_pose_net_fatnet_daspp_dcn_dla_lk_dr,
    'fatnetdasppdcndlalkse': get_pose_net_fatnet_daspp_dcn_dla_lk_se,

    'fatnetfrn': get_pose_net_fatnet_frn,
    'fatnetfrndladaspp': get_pose_net_fatnet_frn_dla_daspp,
    'fatnetfrnmbdasppdcn': get_pose_net_fatnet_frn_mb_daspp_dcn,
    'fatnetfrndladasppdcn':get_pose_net_fatnet_frn_daspp_dcn_dla,
    'fatnetfrndladasppdcnlk':get_pose_net_fatnet_frn_daspp_dcn_dla_lk,

    'fatnetfrndladasppdcnatt':get_pose_net_fatnet_frn_daspp_dcn_dla_att,
    'fatnetfrnpre': get_pose_net_fatnet_frn_branch_daspp_dcn_pretrained,

}


def create_model(arch, heads, head_conv):
    num_layers = int(arch[arch.find('_') + 1:]) if '_' in arch else 0
    arch = arch[:arch.find('_')] if '_' in arch else arch
    get_model = _model_factory[arch]
    model = get_model(num_layers=num_layers, heads=heads, head_conv=head_conv)
    return model


def load_model(model, model_path, optimizer=None, resume=False,
               lr=None, lr_step=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, ' \
                      'loaded shape{}.'.format(
                    k, model_state_dict[k].shape, state_dict[k].shape))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k))
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k))
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for step in lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


def save_model(path, epoch, model, optimizer=None):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    data = {'epoch': epoch,
            'state_dict': state_dict}
    if not (optimizer is None):
        data['optimizer'] = optimizer.state_dict()
    torch.save(data, path)
