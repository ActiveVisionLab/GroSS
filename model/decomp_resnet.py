import torch
import torch.nn as nn
import torch.nn.functional as F

from .decompnet import DecompNet


class DecompResNet(DecompNet):
    def forward(self, x, return_responses=[]):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        # Block 2
        y = self.bn2_1(self.conv2_1(x))
        y = F.relu(y)
        y = self.bn2_2(self.conv2_2(y))
        x = F.relu(x + y)

        y = self.bn2_3(self.conv2_3(x))
        y = F.relu(y)
        y = self.bn2_4(self.conv2_4(y))
        x = F.relu(x + y)

        # Block 3
        y = self.bn3_1(self.conv3_1(x))
        y = F.relu(y)
        y = self.bn3_2(self.conv3_2(y))
        x = self.bn3_d(self.downsample3(x)) + y
        x = F.relu(x)

        y = self.bn3_3(self.conv3_3(x))
        y = F.relu(y)
        y = self.bn3_4(self.conv3_4(y))
        x = F.relu(x + y)

        # Block 4
        y = self.bn4_1(self.conv4_1(x))
        y = F.relu(y)
        y = self.bn4_2(self.conv4_2(y))
        x = self.bn4_d(self.downsample4(x)) + y
        x = F.relu(x)

        y = self.bn4_3(self.conv4_3(x))
        y = F.relu(y)
        y = self.bn4_4(self.conv4_4(y))
        x = F.relu(x + y)

        # Block 5
        y = self.bn5_1(self.conv5_1(x))
        y = F.relu(y)
        y = self.bn5_2(self.conv5_2(y))
        x = self.bn5_d(self.downsample5(x)) + y
        x = F.relu(x)

        y = self.bn5_3(self.conv5_3(x))
        y = F.relu(y)
        y = self.bn5_4(self.conv5_4(y))
        x = F.relu(x + y)

        x = self._forward_classifier(x)

        return x

    def _forward_classifier(self, x):
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def _build_classifier(self):
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, 1000)

    def _build_layers(self):
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1, ceil_mode=False)

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(64)

        self.conv2_3 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2_3 = nn.BatchNorm2d(64)
        self.conv2_4 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=False)
        self.bn2_4 = nn.BatchNorm2d(64)

        # Block 3
        self.conv3_1 = nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.downsample3 = nn.Conv2d(64, 128, 1, stride=2, bias=False)
        self.bn3_d = nn.BatchNorm2d(128)

        self.conv3_3 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn3_3 = nn.BatchNorm2d(128)
        self.conv3_4 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=False)
        self.bn3_4 = nn.BatchNorm2d(128)

        # Block 4
        self.conv4_1 = nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.downsample4 = nn.Conv2d(128, 256, 1, stride=2, bias=False)
        self.bn4_d = nn.BatchNorm2d(256)

        self.conv4_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.bn4_3 = nn.BatchNorm2d(256)
        self.conv4_4 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=False)
        self.bn4_4 = nn.BatchNorm2d(256)

        # Block 5
        self.conv5_1 = nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.downsample5 = nn.Conv2d(256, 512, 1, stride=2, bias=False)
        self.bn5_d = nn.BatchNorm2d(512)

        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False)
        self.bn5_3 = nn.BatchNorm2d(512)
        self.conv5_4 = nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=False)
        self.bn5_4 = nn.BatchNorm2d(512)

        self._build_classifier()
