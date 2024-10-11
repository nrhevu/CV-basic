from __future__ import annotations

import os
from abc import ABC, abstractmethod

import cv2
import numpy as np


class FeatureExtractor(ABC):
    @abstractmethod
    def __call__(self, input: np.ndarray | os.Pathlike):
        pass
    
    def read_image(self, input: np.ndarray | os.PathLike):
        if isinstance(input, os.PathLike):
            return_images = cv2.imread(input)
        elif isinstance(input, str):
            if not os.path.exists(input):
                raise FileNotFoundError(f"File not found: {input}")
            return_images = cv2.imread(input)
        elif isinstance(input, np.ndarray):
            return_images = input
        else:
            raise TypeError(f"Invalid type for image {type(input)}")
                    
        return return_images

    