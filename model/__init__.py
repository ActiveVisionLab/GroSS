from .decompnet import DecompNet
from .decomp_vgg import DecompVGG, DecompVGGImageNet
from .decomp_resnet import DecompResNet

model_factory = {
    '4Conv': DecompNet,
    'VGG': DecompVGG,
    'VGG-ImageNet': DecompVGGImageNet,
    'ResNet18': DecompResNet,
}