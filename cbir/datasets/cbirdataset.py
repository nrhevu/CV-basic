from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from torch.utils.data import Dataset


class CBIRDataset(Dataset, ABC):    
    @abstractmethod
    def __getitem__(self, index) -> Any:
        return super().__getitem__(index)

    @abstractmethod
    def __len__(self) -> int:
        return super().__len__()
    
    @abstractmethod
    def get_by_classes(self, classes: list | str | int) -> Any:
        return super().get_by_classes(classes)