from __future__ import annotations

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from cbir.datasets.cbirdataset import CBIRDataset


class CIFAR100CBIRDaset(CBIRDataset):
    def __init__(
        self, root="./data", train=True, download=True, transform=transforms.ToTensor()
    ):
        self.dataset = torchvision.datasets.CIFAR100(
            root, train=train, download=download, transform=transform
        )

    def __getitem__(self, index) -> torch.Any:
        return self.dataset[index]

    def __len__(self) -> int:
        return len(self.dataset)

    def get_by_classes(self, classes: list | str | int) -> torch.Any:
        if isinstance(classes, list):
            raise NotImplementedError("Not implemented for find by list of classes yet")

        elif isinstance(classes, str):
            classes = self.dataset.class_to_idx[classes]

        return self.__find_item_by_class(classes)

    def __find_item_by_class(self, class_idx: int):
        indexes = np.where(np.array(self.dataset.targets) == class_idx)[0]
        return np.take(self.dataset.data, indexes, axis=0)

    @property
    def classes(self) -> list:
        return self.dataset.classes
