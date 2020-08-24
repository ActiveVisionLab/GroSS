import torch
import torchvision
import torchvision.transforms as transforms


class CIFAR10TrainDataset(torchvision.datasets.CIFAR10):
    train_list = [
        ["data_batch_1", "c99cafc152244af753f735de768cd75f"],
        ["data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"],
        ["data_batch_3", "54ebc095f3ab1f0389bbae665268c751"],
        ["data_batch_4", "634d18415352ddfa80567beed471001a"],
    ]


class CIFAR10ValDataset(torchvision.datasets.CIFAR10):
    train_list = [["data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"]]


dataset_splits = {
    "train": CIFAR10TrainDataset,
    "val": CIFAR10ValDataset,
    "test": torchvision.datasets.CIFAR10,
}


def build_transform(is_train, mean, var):
    transform = [transforms.ToTensor(), transforms.Normalize(mean=mean, std=var)]
    if is_train:
        transform = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Pad(4),
            transforms.RandomCrop(32),
        ] + transform
    transform = transforms.Compose(transform)
    return transform


def build_dataloader(cfg, split="train"):
    root_dir = cfg.DATASET.ROOT_DIR
    batch_size = cfg.TRAIN.BATCH_SIZE if split == "train" else cfg.TEST.BATCH_SIZE
    num_workers = cfg.SYSTEM.NUM_WORKERS

    norm_mean = cfg.DATASET.NORM_MEAN
    norm_var = cfg.DATASET.NORM_VAR

    is_train = split == "train"
    transform = build_transform(is_train, norm_mean, norm_var)
    dataset = dataset_splits[split](root=root_dir, train=is_train, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers
    )
    return dataloader
