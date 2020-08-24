import random
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import choice
from tqdm import tqdm

from engine.btd import pci


def decompose_weight(cfg, weight, num_groups, bottle_dim, remainders=[]):
    o, i, k1, k2 = weight.shape
    weight = weight.view(o, i, -1).permute(1, 0, 2).contiguous()

    # Decompose the layer
    cores_and_factors, remainder = pci(cfg, weight, num_groups, bottle_dim, remainders)

    # Stack the decomp components into tensors
    g_w = []
    p1_w = []
    p2_w = []
    for c, f in cores_and_factors:
        a, b = f
        g_w.append(c.transpose(0, 1))
        p1_w.append(a.transpose(0, 1))
        p2_w.append(b)
    g_w = torch.cat(g_w, dim=0).view(bottle_dim[1], -1, k1, k2)
    p1_w = torch.cat(p1_w, dim=0).unsqueeze(-1).unsqueeze(-1)
    p2_w = torch.cat(p2_w, dim=1).unsqueeze(-1).unsqueeze(-1)

    return p1_w, g_w, p2_w


def expand_and_split_weights(p1, g, p2, target_group_size):
    s, in_channels = p1.shape[:2]
    out_channels, t = p2.shape[:2]
    origin_group_size = g.shape[1]
    k1, k2 = g.shape[-2:]
    assert origin_group_size < target_group_size
    # expand g to the new group size
    g = expand_group_conv(g.detach(), origin_group_size, target_group_size, s)
    # reverse the c + f -> weight process in decompose_weight
    # arrage to correct shapes
    p1, p2 = p1.detach().squeeze(), p2.detach().squeeze()
    g = g.view(t, target_group_size, -1)
    # split into the correct cores and factors
    num_groups = s // target_group_size
    a = torch.chunk(p1, num_groups, dim=0)
    b = torch.chunk(p2, num_groups, dim=1)
    c = torch.chunk(g, num_groups, dim=0)
    # transpose a and c
    a = [factor.transpose(0, 1) for factor in a]
    c = [core.transpose(0, 1) for core in c]
    return a, c, b


def expand_group_conv(weight, origin_group_size, target_group_size, in_channels=None):
    if origin_group_size == target_group_size:
        return weight
    out_channels, _, k1, k2 = weight.shape
    # unless given we assume that in_channels = out_channels
    if in_channels is None:
        in_channels = out_channels
    # pattern is depends on the number of filters per group
    num_groups = in_channels // origin_group_size
    filters_per_group = out_channels // num_groups
    split_filters = torch.split(weight, filters_per_group, 0)

    exp_weight = []
    for i, split in enumerate(split_filters):
        to_cat = []
        n, og, k1, k2 = split.shape
        n_pre = (i * origin_group_size) % target_group_size
        if n_pre > 0:
            to_cat.append(split.new_zeros(n, n_pre, k1, k2))
        to_cat.append(split)
        n_post = (target_group_size - (i + 1) * origin_group_size) % target_group_size
        if n_post > 0:
            to_cat.append(split.new_zeros(n, n_post, k1, k2))
        exp_weight.append(torch.cat(to_cat, dim=1))
    exp_weight = torch.cat(exp_weight)
    return exp_weight


class SeriesDecompConv(nn.Module):
    def __init__(self, group_sizes, layer, decompose=True, cfg=None, bottle_dim=-1):
        super(SeriesDecompConv, self).__init__()
        in_channels = layer.weight.shape[1]
        if not isinstance(bottle_dim, list):
            self.bottle_dim = (
                min(bottle_dim, in_channels) if bottle_dim != -1 else in_channels
            )
        else:
            self.bottle_dim = bottle_dim
        # perform series decomp of layer
        if decompose:
            assert cfg is not None
            layer_weights = self._decompose(cfg, layer.weight.detach(), group_sizes)
        else:
            layer_weights = self._init_weights(layer.weight, group_sizes, bottle_dim)

        self.layer_weights = nn.ModuleList(
            [
                nn.ParameterList([nn.Parameter(p1), nn.Parameter(g), nn.Parameter(p2)])
                for p1, g, p2 in layer_weights
            ]
        )
        if layer.bias is not None:
            self.bias = nn.Parameter(layer.bias)
        else:
            self.bias = None
        self.stride = layer.stride
        self.padding = layer.padding
        self.in_channels = layer.weight.shape[1]
        self.out_channels = layer.weight.shape[0]
        self.kernel_size = layer.weight.shape[2]
        self.group_sizes = group_sizes
        self.train_gs = -1
        self.test_gs = self.in_channels

    def forward(self, x):
        group_size = self.train_gs if self.training else self.test_gs
        if group_size < 0:
            group_size = self._sample_group_size()
        if isinstance(self.bottle_dim, list):
            num_groups = self.bottle_dim[0] // group_size
        else:
            num_groups = self.bottle_dim // group_size
        p1_w, g_w, p2_w = self._make_layer(group_size)
        x = F.conv2d(x, p1_w, stride=self.stride)
        x = F.conv2d(x, g_w, groups=num_groups, padding=self.padding)
        x = F.conv2d(x, p2_w, bias=self.bias)
        return x

    def fuse(self):
        self.train_gs = self.test_gs

        p1, g, p2 = self._make_layer(self.test_gs)
        self.layer_weights = nn.ModuleList(
            [nn.ParameterList([nn.Parameter(p1), nn.Parameter(g), nn.Parameter(p2)])]
        )
        self.group_sizes = [self.test_gs]

    def _decompose(self, cfg, weight, group_sizes):
        # run through the group sizes in ascending order
        if isinstance(self.bottle_dim, list):
            s, t = self.bottle_dim
        else:
            s = self.bottle_dim
            t = self.bottle_dim
        decomp_weights = []
        group_sizes.sort()
        for size in group_sizes:
            num_groups = s // size
            rf1, rc, rf2 = list(), list(), list()
            with torch.no_grad():
                # loop through group sizes that have already been decomposed
                for gs, (p1, g, p2) in tqdm(zip(group_sizes, decomp_weights)):
                    f1, c, f2 = expand_and_split_weights(p1, g, p2, size)
                    rf1.append(f1)
                    rc.append(c)
                    rf2.append(f2)

                if rc and rf1 and rf2:
                    rc = list(map(sum, zip(*rc)))
                    rf1 = list(map(sum, zip(*rf1)))
                    rf2 = list(map(sum, zip(*rf2)))
                    remainders = list(zip(*(rf1, rc, rf2)))
                else:
                    remainders = []
            weights = decompose_weight(cfg, weight, num_groups, (s, t), remainders)
            decomp_weights.append(weights)
        return decomp_weights

    def _init_weights(self, weight, group_sizes, bottle_dim=-1):
        o, i, k1, k2 = weight.shape
        if isinstance(self.bottle_dim, list):
            s, t = self.bottle_dim
        else:
            s = self.bottle_dim if self.bottle_dim > 0 else i
            t = self.bottle_dim if self.bottle_dim > 0 else i
        weights = []
        for size in group_sizes:
            p1 = nn.init.normal_(weight.new_zeros((s, i, 1, 1)))
            p2 = nn.init.normal_(weight.new_zeros((o, t, 1, 1)))
            g = nn.init.normal_(weight.new_zeros((t, size, k1, k2)))
            weights.append((p1, g, p2))
        return weights

    def _make_layer(self, group_size):
        point1 = []
        group = []
        point2 = []

        if isinstance(self.bottle_dim, list):
            s, t = self.bottle_dim
        else:
            s = self.bottle_dim
            t = self.bottle_dim

        for i, (gs, (p1, g, p2)) in enumerate(
            zip(self.group_sizes, self.layer_weights)
        ):
            if gs > group_size:
                break
            point1.append(p1)
            group.append(expand_group_conv(g, gs, group_size, s))
            point2.append(p2)
        point1 = sum(point1)
        group = sum(group)
        point2 = sum(point2)

        return point1.contiguous(), group.contiguous(), point2.contiguous()

    def _sample_group_size(self):
        return int(choice(self.group_sizes))
