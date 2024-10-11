from __future__ import annotations

from abc import ABC, abstractmethod
import os

import numpy as np

class ImageSearchObject():
    def __init__(self, image: np.ndarray, score: float) -> None:
        self.image = image
        self.score = score

class CBIR(ABC):
    @abstractmethod
    def fit(self, images: list) -> None:
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray | os.PathLike) -> list[ImageSearchObject]:
        pass