from __future__ import annotations

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models

from cbir.feature_extractor import BatchFeatureExtractor


class VGG16Extractor(BatchFeatureExtractor):
    def __init__(self, device: str = "cpu") -> None:
        VGG = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.device = device
        self.pretrained_model = VGG
        self.pretrained_model.classifier = nn.Sequential(*list(VGG.classifier.children())[:-3])
        self.pretrained_model = self.pretrained_model.eval()
        self.pretrained_model = self.pretrained_model.to(self.device)
        
    def __call__(self, input: np.ndarray | list[os.PathLike]):
        imgs = super().read_image(input)
        
        with torch.no_grad():
            imgs = torch.from_numpy(imgs).float().to(self.device)
            output = self.pretrained_model(imgs)
        feature = output.data.cpu().numpy()
        feature_norm = feature/np.linalg.norm(feature)
        # out /= out.sum(axis=1, keepdims=True)
        return feature_norm