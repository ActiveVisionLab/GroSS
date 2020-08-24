from tqdm import tqdm, trange
import argparse

from config import get_cfg_defaults
from model import model_factory
from engine.train import train_on_dataset

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
    parser.add_argument(
        "-r",
        "--resume-file",
        default=None,
        metavar="FILE",
        help="path to checkpoint from which to resume",
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

    for r in trange(cfg.TRAIN.NUM_RUNS):
        if cfg.MODEL.RUN_SPECIFIC_LOAD_PATH:
            load_path = cfg.MODEL.LOAD_PATH.format(r)
        else:
            load_path = None
        model = model_factory[cfg.MODEL.BACKBONE](cfg, load_path).cuda()
        model = model.decompose(cfg)
        if cfg.MODEL.FUSE_FOR_TRAIN:
            model = model.fuse()
        train_on_dataset(cfg, model, r, resume_path=args.resume_file)
