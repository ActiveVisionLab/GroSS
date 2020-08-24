import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm, trange

from config import get_cfg_defaults
from dataset import dataloader_factory
from model import model_factory
from model.decompnet import freeze_non_decomposed_layers
from utils.lr_scheduler import WarmupMultiStepLR

from .eval import eval_on_dataset


def write_summary(writer, summary, iter_num):
    for label, value in summary.items():
        writer.add_scalar(label, value, iter_num)


def save_checkpoint(model, save_path, epoch, optimiser, scheduler):
    if isinstance(model, nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    save_dict = {
        "model": state_dict,
        "epoch": epoch,
        "optimiser": optimiser,
        "scheduler": scheduler,
    }
    torch.save(save_dict, save_path)


def load_checkpoint(model, load_path, optimiser, scheduler):
    load_dict = torch.load(load_path)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(load_dict["model"])
    else:
        model.load_state_dict(load_dict["model"])
    optimiser.load_state_dict(load_dict["optimiser"])
    scheduler.load_state_dict(load_dict["scheduler"])
    epoch = load_dict["epoch"]
    return model, epoch, optimiser, scheduler


def train_on_dataset(cfg, model, run_number, resume_path=None):
    torch.backends.cudnn.benchmark = True
    out_dir = os.path.join(cfg.OUT_DIR, str(run_number))

    train_loader = dataloader_factory[cfg.DATASET.NAME](cfg, split="train")
    test_loader = dataloader_factory[cfg.DATASET.NAME](cfg, split="val")

    writer = SummaryWriter(log_dir=out_dir)
    loss_label = cfg.TRAIN.LOSS_LABEL
    acc_label = cfg.TRAIN.ACC_LABEL
    save_every = cfg.TRAIN.SAVE_EVERY
    test_every = cfg.TRAIN.TEST_EVERY
    early_stopping = cfg.TRAIN.EARLY_STOPPING
    epoch_size = (
        cfg.TRAIN.EPOCH_SIZE if cfg.TRAIN.FIXED_EPOCH_SIZE else len(train_loader)
    )

    # Configure and save model
    if cfg.TRAIN.FREEZE_STANDARD_LAYERS:
        model = freeze_non_decomposed_layers(model)
    for layer in model.modules():
        if isinstance(layer, nn.BatchNorm2d):
            if cfg.TRAIN.FREEZE_BN_LAYERS:
                layer.eval()
                layer.weight.requires_grad = False
                layer.bias.requires_grad = False
    if cfg.TRAIN.SAVE_AT_START:
        save_path = os.path.join(out_dir, "decompnet_{}.pth".format(0))
        torch.save(model.state_dict(), save_path)

    # Loss and schedule
    criterion = nn.CrossEntropyLoss()
    optimiser = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.TRAIN.INITIAL_LR,
        momentum=cfg.TRAIN.MOMENTUM,
        weight_decay=cfg.TRAIN.WEIGHT_DECAY,
    )
    scheduler = WarmupMultiStepLR(
        optimiser,
        milestones=cfg.TRAIN.LR_MILESTONES,
        gamma=cfg.TRAIN.LR_GAMMA,
        warmup_factor=cfg.TRAIN.LR_WARMUP_FACTOR,
        warmup_iters=cfg.TRAIN.LR_WARMUP_EPOCHS,
    )

    if resume_path is not None:
        model, start_epoch, optimiser, scheduler = load_checkpoint(
            model, resume_path, optimiser, scheduler
        )
        start_epoch += 1
    else:
        start_epoch = 0

    # Parallelise the model if there is more than 1 GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    accuracy = 0
    if early_stopping:
        highest_accuracy = 0
    for epoch in trange(start_epoch, cfg.TRAIN.NUM_EPOCHS):
        model = model.train()
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                if cfg.TRAIN.FREEZE_BN_LAYERS:
                    layer.eval()
                    layer.weight.requires_grad = False
                    layer.bias.requires_grad = False

        with tqdm(enumerate(train_loader)) as t:
            for i, data in t:
                if i == epoch_size:
                    break
                iter_num = i + epoch * epoch_size
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                optimiser.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimiser.step()

                summary_dict = {loss_label: loss.item()}
                write_summary(writer, summary_dict, iter_num)
                for p in optimiser.param_groups:
                    lr = p["lr"]
                t.set_postfix(loss=loss.item(), lr=lr, epoch=epoch, prev_acc=accuracy)
        scheduler.step()

        if epoch % test_every == (test_every - 1):
            accuracy = eval_on_dataset(model, test_loader)
            writer.add_scalar("{}".format(acc_label), accuracy, epoch + 1)
            if early_stopping:
                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    save_path = os.path.join(out_dir, "decompnet_high_acc.pth")
                    save_checkpoint(model, save_path, epoch, optimiser, scheduler)
            # Perform additional testing if defined in config
            if cfg.TEST.ADDITIONAL_TEST:
                add_group_sizes = cfg.TEST.ADDITIONAL_TEST_GROUP_SIZES
                if isinstance(model, nn.DataParallel):
                    prev_group_sizes = model.module.test_group_sizes
                    model.module.test_group_sizes = add_group_sizes
                else:
                    prev_group_sizes = model.test_group_sizes
                    model.test_group_sizes = add_group_sizes
                add_accuracy = eval_on_dataset(model, test_loader)
                writer.add_scalar(
                    "{}".format(cfg.TEST.ADDITIONAL_ACC_LABEL), add_accuracy, epoch + 1
                )
                if isinstance(model, nn.DataParallel):
                    model.module.test_group_sizes = prev_group_sizes
                else:
                    model.test_group_sizes = prev_group_sizes

        if epoch % save_every == (save_every - 1):
            save_path = os.path.join(out_dir, "decompnet_{}.pth".format(epoch + 1))
            save_checkpoint(model, save_path, epoch, optimiser, scheduler)
