from tqdm import trange
import argparse

from config import get_cfg_defaults
from dataset.cifar10 import build_dataloader
from engine.train import train_on_dataset
from engine.eval import eval_on_dataset
from model import model_factory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    for r in trange(0, cfg.TRAIN.NUM_RUNS):
        model = model_factory[cfg.MODEL.BACKBONE](cfg).cuda()
        train_on_dataset(cfg, model, r)
