import torch
import torch.nn as nn
from tqdm import tqdm


def eval_on_dataset(model, test_loader, return_top5=False):
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1 and not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model = model.eval()
    correct = 0
    total = 0
    if return_top5:
        correct_5 = 0
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            images = images.cuda().contiguous()
            outputs = model(images.cuda()).cpu().detach()
            _, predicted = torch.max(outputs.data, 1)
            if return_top5:
                _, predicted_5 = torch.topk(outputs.data, k=5, dim=1)
                for p5, l in zip(predicted_5, labels):
                    if l in p5:
                        correct_5 += 1
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = correct / total
    model = model.train()
    if return_top5:
        accuracy_5 = correct_5 / total
        return accuracy, accuracy_5
    return accuracy


def eval_multiple_configs(model, test_loader, configs):
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.eval()
    correct = [0] * len(configs)
    total = [0] * len(configs)
    with torch.no_grad():
        for data in tqdm(test_loader):
            images, labels = data
            images = images.cuda()
            for i, config in enumerate(configs):
                if torch.cuda.device_count() > 1:
                    model.module.test_group_sizes = config
                else:
                    model.test_group_sizes = config
                outputs = model(images).cpu().detach()
                _, predicted = torch.max(outputs.data, 1)
                total[i] += labels.size(0)
                correct[i] += (predicted == labels).sum().item()
        accuracy = [c / t for c, t in zip(correct, total)]
    model = model.train()
    return accuracy
