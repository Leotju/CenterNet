import torch
from torch import nn
from torch.nn.functional import interpolate


class GloRe(nn.Module):
    def __init__(self, in_channels):
        super(GloRe, self).__init__()
        self.N = in_channels // 4
        self.S = in_channels // 2

        self.theta = nn.Conv2d(in_channels, self.N, 1, 1, 0, bias=False)
        self.phi = nn.Conv2d(in_channels, self.S, 1, 1, 0, bias=False)

        self.relu = nn.ReLU()
        self.node_conv = nn.Conv1d(self.N, self.N, 1, 1, 0, bias=False)
        self.channel_conv = nn.Conv1d(self.S, self.S, 1, 1, 0, bias=False)
        self.conv_2 = nn.Conv2d(self.S, in_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        batch, C, H, W = x.size()
        L = H * W
        B = self.theta(x).view(-1, self.N, L)
        phi = self.phi(x).view(-1, self.S, L)
        phi = torch.transpose(phi, 1, 2)
        V = torch.bmm(B, phi) / L
        V = self.relu(self.node_conv(V))
        V = self.relu(self.channel_conv(torch.transpose(V, 1, 2)))
        y = torch.bmm(torch.transpose(B, 1, 2), torch.transpose(V, 1, 2))
        y = y.view(-1, self.S, H, W)
        y = self.conv_2(y)
        return x + y
