import torch
from torch import nn
from torch.nn.functional import interpolate


class GloReLeo(nn.Module):
    def __init__(self,
                 in_channels,
                 num_feats,
                 reduce_channels,
                 use_scale=True):
        super(GloReLeo, self).__init__()
        self.in_channels = in_channels
        self.use_scale = use_scale
        self.reduce_channels = reduce_channels
        self.num_feats = num_feats

        self.phi = nn.Linear(in_channels, reduce_channels)
        self.theta = nn.Linear(in_channels, num_feats)
        self.ag = nn.Conv1d(num_feats, num_feats, 1)
        self.wg = nn.Conv1d(reduce_channels, reduce_channels, 1)
        self.conv_out = nn.Linear(reduce_channels, in_channels)

        self.init_weights()

    def init_weights(self, std=0.01, zeros_init=True):
        # pass
        nn.init.xavier_uniform_(self.phi.weight)
        nn.init.constant_(self.phi.bias, 0)
        nn.init.xavier_uniform_(self.theta.weight)
        nn.init.constant_(self.theta.bias, 0)


    def forward(self, x):
        bt, c, h, w = x.shape
        identity = x
        x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

        # B  x C x N
        xr = self.phi(x)
        # B x Cr x N
        b = self.theta(x).permute(0, 2, 1)
        # B x N x Cr
        v = torch.matmul(b, xr)
        # B x N x N
        Ag = self.ag(v).sigmoid()
        # B x N x N
        y = self.wg((v - Ag).permute(0, 2, 1)).permute(0, 2, 1)
        # B x N x N
        out = torch.matmul(b.permute(0, 2, 1), y)
        # B x Cr x N
        out = self.conv_out(out)
        # B x C x N
        output = identity + out.permute(0, 2, 1).view(bt, c, h, w)

        return output
