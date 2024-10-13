from __future__ import annotations, print_function

from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import models

from cbir.feature_extractor import BatchFeatureExtractor


class ResidualNet(nn.Module):
    def __init__(self, model="resnet152", pretrained=True):
        super(ResidualNet, self).__init__()
        if model == "resnet18":
            if pretrained:
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model == "resnet34":
            if pretrained:
                self.model = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        elif model == "resnet50":
            if pretrained:
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        elif model == "resnet101":
            if pretrained:
                self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
        elif model == "resnet152":
            if pretrained:
                self.model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
    
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)  # x after layer4, shape = N * 512 * H/32 * W/32
        avg = self.model.avgpool(x)
        avg = avg.view(avg.size(0), -1)
        return avg

    
class ResNetExtractor(BatchFeatureExtractor):
    def __init__(self, model: str, device: str = "cpu") -> None:
        self.model = ResidualNet(model, pretrained=True)
        self.device = device
        self.model.model.to(device)
        self.model.model.eval()
        
    
    def __call__(self, input: np.ndarray | torch.Tensor | Any):
        imgs = super().read_image(input)
        with torch.no_grad():
            imgs = Variable(torch.from_numpy(imgs).float()).to(self.device)
            output = self.model(imgs)
        out = output.cpu().numpy()
        # out /= out.sum(axis=1, keepdims=True)
        return out
        
        
        