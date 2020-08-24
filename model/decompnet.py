import os

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .series_decomp import SeriesDecompConv


def pow_2_up_2_n(n):
    """ :) """
    powers = []
    i = 0
    while 2 ** i <= n:
        powers.append(2 ** i)
        i += 1
    return powers


def freeze_non_decomposed_layers(model):
    for layer in model.children():
        # BatchNorms are handled explicitly in the train loop
        if not isinstance(layer, SeriesDecompConv) and not isinstance(
            layer, nn.BatchNorm2d
        ):
            for p in layer.parameters():
                p.requires_grad = False
        else:
            if layer.bias is not None:
                layer.bias.requires_grad = False
    return model


class DecompNet(nn.Module):
    def __init__(self, cfg, load_path=None, load_model=True):
        super(DecompNet, self).__init__()
        self._build_layers()
        self._init_weights()

        self.decomposable_layers = cfg.MODEL.DECOMPOSABLE_LAYERS
        self.decompose_group_sizes = cfg.MODEL.DECOMPOSE_STRUCTURE
        self.bottle_dims = cfg.MODEL.BOTTLE_DIMS
        self._train_group_sizes = cfg.TRAIN.GROUP_SIZES
        self._test_group_sizes = cfg.TEST.GROUP_SIZES
        self.cfg = cfg.clone()

        # load file from the config can be overridden
        load_path = cfg.MODEL.LOAD_PATH if load_path is None else load_path
        for layer_name in self.decomposable_layers:
            layer = getattr(self, layer_name)
            layer.is_decomposed = False
            setattr(self, layer_name, layer)
        if cfg.MODEL.INIT_FUSED:
            self._init_decomposed(cfg)
            self.fuse()
            self.load_weights(load_path)
        elif cfg.MODEL.INIT_DECOMPOSED:
            self._init_decomposed(cfg, load_path, load_model)
        elif load_model:
            self.load_weights(load_path)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(F.relu(x))

        x = self.conv2(x)
        x = self.pool2(F.relu(x))

        x = self.conv3(x)
        x = self.pool3(F.relu(x))

        x = self.conv4(x)
        x = self.pool4(F.relu(x))

        x = F.relu(self.fc1(x.view(-1, 4 * 64)))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def decompose(self, cfg):
        self._decompose_network(perform_btd=True, cfg=cfg)
        return self

    def fuse(self):
        self._fuse_network()
        return self

    def load_weights(self, load_path="", verbose=True):
        if len(load_path) != 0:
            load_dict = torch.load(load_path)
            if "optimiser" in load_dict.keys():
                load_dict = load_dict["model"]
            if verbose:
                print("loading model from: {}".format(load_path))
            self.load_state_dict(load_dict)

    @property
    def train_group_sizes(self):
        return self._train_group_sizes

    @train_group_sizes.setter
    def train_group_sizes(self, group_sizes):
        self._train_group_sizes = group_sizes
        for layer_name, group_size in zip(self.decomposable_layers, group_sizes):
            layer = getattr(self, layer_name)
            if isinstance(layer, SeriesDecompConv):
                layer.train_gs = group_size
            setattr(self, layer_name, layer)

    @property
    def test_group_sizes(self):
        return self._test_group_sizes

    @test_group_sizes.setter
    def test_group_sizes(self, group_sizes):
        self._test_group_sizes = group_sizes
        for layer_name, group_size in zip(self.decomposable_layers, group_sizes):
            layer = getattr(self, layer_name)
            if isinstance(layer, SeriesDecompConv):
                layer.test_gs = group_size
            setattr(self, layer_name, layer)

    def _build_layers(self):
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
        self.pool4 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(4 * 64, 256)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(256, 10)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _init_decomposed(self, cfg=None, load_path="", load_model=True):
        self._decompose_network(perform_btd=False, cfg=cfg)
        if load_model:
            self.load_weights(load_path)

    def _decompose_network(self, perform_btd=True, cfg=None):
        self.save_name = []
        if self.bottle_dims:
            layer_decomp_args = [
                (cfg, perform_btd, layer_name, layer_gs, train_gs, test_gs, bottle_dim)
                for layer_name, layer_gs, train_gs, test_gs, bottle_dim in zip(
                    self.decomposable_layers,
                    self.decompose_group_sizes,
                    self.train_group_sizes,
                    self.test_group_sizes,
                    self.bottle_dims,
                )
            ]
        else:
            layer_decomp_args = [
                (cfg, perform_btd, layer_name, layer_gs, train_gs, test_gs)
                for layer_name, layer_gs, train_gs, test_gs in zip(
                    self.decomposable_layers,
                    self.decompose_group_sizes,
                    self.train_group_sizes,
                    self.test_group_sizes,
                )
            ]
        for decomp_args in tqdm(layer_decomp_args):
            self._decompose_layer(*decomp_args)

    def _decompose_layer(
        self, cfg, perform_btd, layer_name, layer_gs, train_gs, test_gs, bottle_dim=-1
    ):
        layer = getattr(self, layer_name)
        if layer.is_decomposed:
            return
        in_channels = layer.weight.shape[1]
        if not isinstance(bottle_dim, list):
            bottle_dim = (
                min(bottle_dim, in_channels) if bottle_dim != -1 else in_channels
            )
        group_sizes = layer_gs if layer_gs else pow_2_up_2_n(bottle_dim)
        decomp_layer = SeriesDecompConv(
            group_sizes, layer, decompose=perform_btd, cfg=cfg, bottle_dim=bottle_dim
        )
        decomp_layer.train_gs = train_gs
        decomp_layer.test_gs = test_gs
        decomp_layer.is_decomposed = True
        decomp_layer.bottle_dim = bottle_dim
        setattr(self, layer_name, decomp_layer)

    def _freeze_group_sizes(self):
        for layer_name, group_sizes in zip(
            self.decomposable_layers, self._frozen_group_sizes
        ):
            layer = getattr(self, layer_name)
            if isinstance(layer, SeriesDecompConv):
                layer.frozen_group_sizes = group_sizes
            setattr(self, layer_name, layer)

    def _fuse_network(self):
        for layer_name in self.decomposable_layers:
            self._fuse_layer(layer_name)

    def _fuse_layer(self, layer_name):
        layer = getattr(self, layer_name)
        if not layer.is_decomposed:
            return
        layer.fuse()
        setattr(self, layer_name, layer)

