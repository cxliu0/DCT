import torch
import torch.nn as nn

from torchvision import models

class Conhead(nn.Module):
    def __init__(self):
        super(Conhead, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, 6)

    def forward(self, x):
        x = self.resnet18(x)
        return x 