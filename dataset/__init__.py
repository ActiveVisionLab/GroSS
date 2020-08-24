from .cifar10 import build_dataloader as cifar_dataloader
from .imagenet import build_dataloader as imagenet_dataloader

dataloader_factory = {
    "cifar10": cifar_dataloader,
    "imagenet": imagenet_dataloader
}