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
                self.model = models.resnet18(pretrained=True)
        elif model == "resnet34":
            if pretrained:
                self.model = models.resnet34(pretrained=True)
        elif model == "resnet50":
            if pretrained:
                self.model = models.resnet50(pretrained=True)
        elif model == "resnet101":
            if pretrained:
                self.model = models.resnet101(pretrained=True)
        elif model == "resnet152":
            if pretrained:
                self.model = models.resnet152(pretrained=True)
    
    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)  # x after layer4, shape = N * 512 * H/32 * W/32
        max_pool = torch.nn.MaxPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False)
        Max = max_pool(x)  # avg.size = N * 512 * 1 * 1
        Max = Max.view(Max.size(0), -1)  # avg.size = N * 512
        avg_pool = torch.nn.AvgPool2d((x.size(-2),x.size(-1)), stride=(x.size(-2),x.size(-1)), padding=0, ceil_mode=False, count_include_pad=True)
        avg = avg_pool(x)  # avg.size = N * 512 * 1 * 1
        avg = avg.view(avg.size(0), -1)  # avg.size = N * 512
        fc = self.model.fc(avg)  # fc.size = N * 1000
        output = {
            'max': Max,
            'avg': avg,
            'fc' : fc
        }
        return output

    
class ResNetExtractor(BatchFeatureExtractor):
    def __init__(self, model: str, pick_layer : Literal["max", "avg", "fc"], device: str = "cpu") -> None:
        self.model = ResidualNet(model, pretrained=True)
        self.pick_layer = pick_layer
        self.device = device
        self.model.model.to(device)
        self.model.model.eval()
        
    
    def __call__(self, input: np.ndarray | torch.Tensor | Any):
        imgs = super().read_image(input)
        with torch.no_grad():
            imgs = Variable(torch.from_numpy(imgs).float()).to(self.device)
            output = self.model(imgs)
        return output[self.pick_layer].cpu().numpy()
        
        
        