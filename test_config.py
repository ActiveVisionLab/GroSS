import argparse
import os

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm, trange

from config import get_cfg_defaults
from dataset import dataloader_factory
from engine.eval import eval_on_dataset
from model import model_factory
from utils.thop import count_macs


def generate_load_paths(cfg):
    paths = []
    for i in range(cfg.TRAIN.NUM_RUNS):
        if cfg.TRAIN.EARLY_STOPPING:
            path = os.path.join(cfg.OUT_DIR, str(i), "decompnet_high_acc.pth")
        else:
            path = os.path.join(
                cfg.OUT_DIR, str(i), "decompnet_{}.pth".format(cfg.TRAIN.NUM_EPOCHS)
            )
        paths.append(path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)

    opts = ["MODEL.INIT_DECOMPOSED", True]
    if cfg.MODEL.FUSE_FOR_TRAIN:
        opts.extend(["MODEL.INIT_FUSED", True])
    cfg.merge_from_list(opts)
    cfg.freeze()

    test_loader = dataloader_factory[cfg.DATASET.NAME](cfg, split="val")

    macs = []
    accs = []
    paths = generate_load_paths(cfg)
    for path in paths:
        model = model_factory[cfg.MODEL.BACKBONE](cfg, path).cuda()
        macs.append(count_macs(model, cfg.DATASET.INPUT_SIZE))
        acc, acc5 = eval_on_dataset(model, test_loader, return_top5=True)
        accs.append(acc)

    print(macs[0])
    print(np.mean(accs), np.std(accs))
