from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np


class Retrieve(ABC):
    @abstractmethod
    def fit(self, features: list, indexes: list) -> None:
        assert len(features) == len(indexes), \
            "Features and indexes must have same length"

    @abstractmethod
    def predict(
        self, features: np.ndarray | list[np.ndarray], k=5
    ) -> tuple[list, list]:
        pass
