import os

import torch
import torchvision
import torchvision.transforms as transforms

def monkey_load(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (sample, target) where target is class_index of the target class.
    """
    path, target = self.samples[index]
    sample = self.loader(path)
    if self.transform is not None:
        sample = self.transform(sample)
    if self.target_transform is not None:
        target = self.target_transform(target)

    return sample, target

def build_transform(is_train, mean, var):
    transform = (
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=var),
        ]
        if is_train
        else [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=var),
        ]
    )
    transform = transforms.Compose(transform)
    return transform


def build_dataloader(cfg, split="train"):
    is_train = split == "train"
    # Use val split for test as mentioned in paper
    if split == "test":
        split = "val"

    root_dir = cfg.DATASET.ROOT_DIR
    batch_size = cfg.TRAIN.BATCH_SIZE if is_train else cfg.TEST.BATCH_SIZE
    num_workers = cfg.SYSTEM.NUM_WORKERS

    norm_mean = cfg.DATASET.NORM_MEAN
    norm_var = cfg.DATASET.NORM_VAR

    
    transform = build_transform(is_train, norm_mean, norm_var)
    dataset = torchvision.datasets.ImageNet(
        root=root_dir, split=split, transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers
    )
    return dataloader
