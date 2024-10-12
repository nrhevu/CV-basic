from __future__ import annotations

from abc import ABC, abstractmethod
import os
from typing import List

import numpy as np
from cbir import FeatureExtractor, FeatureStore
class ImageSearchObject():
    def __init__(self, image: np.ndarray, score: float) -> None:
        self.image = image
        self.score = score

class CBIR():
    def __init__(self, feature_extractor : FeatureExtractor, feature_store : FeatureStore):
        self.feature_extractor = feature_extractor
        self.feature_store = feature_store
    @abstractmethod
    def indexing(self, images: List[np.ndarray | os.PathLike] | np.ndarray | os.PathLike) -> None:
        pass
    
    @abstractmethod
    def retrieve(self, image: np.ndarray | os.PathLike) -> list[ImageSearchObject]:
        pass