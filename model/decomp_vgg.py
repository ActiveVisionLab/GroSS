import torch.nn as nn
import torch.nn.functional as F

from .decompnet import DecompNet

class DecompVGG(DecompNet):
    def forward(self, x):
        responses = []
        x = self.conv1_1(x)
        x = self.conv1_2(F.relu(x))
        x = self.pool1(F.relu(x))

        x = self.conv2_1(x)
        x = self.conv2_2(F.relu(x))
        x = self.pool2(F.relu(x))

        x = self.conv3_1(x)
        x = self.conv3_2(F.relu(x))
        x = self.conv3_3(F.relu(x))
        x = self.pool3(F.relu(x))

        x = self.conv4_1(x)
        x = self.conv4_2(F.relu(x))
        x = self.conv4_3(F.relu(x))
        x = self.pool4(F.relu(x))

        x = self.conv5_1(x)
        x = self.conv5_2(F.relu(x))
        x = self.conv5_3(F.relu(x))
        x = self.pool5(F.relu(x))

        x = self._forward_classifier(x)
        return x

    def _forward_classifier(self, x):
        x = F.relu(self.fc1(x.view(-1, 512)))
        x = self.drop1(x)
        x = self.fc2(x)
        return x

    def _build_classifier(self):
        self.fc1 = nn.Linear(512, 512)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(512, 10)

    def _build_layers(self):
        self.conv1_1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, stride=1, padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, stride=1, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, stride=1, padding=1, bias=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, stride=1, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4_1 = nn.Conv2d(256, 512, 3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
        self.pool4 = nn.MaxPool2d(2)

        self.conv5_1 = nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, stride=1, padding=1, bias=True)
        self.pool5 = nn.MaxPool2d(2)

        self._build_classifier()


class DecompVGGImageNet(DecompVGG):
    def _forward_classifier(self, x):
        x = F.relu(self.fc1(x.view(-1, 7 * 7 * 512)))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc3(x)
        return x

    def _build_classifier(self):
        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(4096, 1000)