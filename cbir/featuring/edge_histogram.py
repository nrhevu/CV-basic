from __future__ import annotations

import itertools
import os
from math import sqrt
from typing import Literal

import cv2
import numpy as np

from cbir.feature_extractor import (BatchFeatureExtractor, FeatureExtractor,
                                    SingleFeatureExtractor)


class EdgeHistogram(SingleFeatureExtractor):
    def __init__(
        self,
        n_slice=10,
        h_type="region",
        bin=16,
        edge_detector: Literal["canny", "sobel", "prewitt"] = "canny",
        normalize=True,
        **kwargs,
    ) -> None:
        self.n_slice = n_slice
        self.h_type = h_type
        self.bin = bin
        self.normalize = normalize
        if edge_detector == "sobel":
            self.ksize = kwargs.get("ksize", 3)
        self.edge_detector = self._get_edge_detector(edge_detector)

    def __call__(self, input: np.ndarray | os.PathLike):
        img = super().read_image(input)

        height, width, channel = img.shape

        if self.h_type == "global":
            hist = self._conv(img)

        elif self.h_type == "region":
            hist = np.zeros((self.n_slice, self.n_slice, self.bin))
            h_silce = np.around(
                np.linspace(0, height, self.n_slice + 1, endpoint=True)
            ).astype(int)
            w_slice = np.around(
                np.linspace(0, width, self.n_slice + 1, endpoint=True)
            ).astype(int)

            for hs in range(len(h_silce) - 1):
                for ws in range(len(w_slice) - 1):
                    img_r = img[
                        h_silce[hs] : h_silce[hs + 1], w_slice[ws] : w_slice[ws + 1]
                    ]  # slice img to regions
                    hist[hs][ws] = self._conv(img_r)

        if self.normalize:
            hist /= np.sum(hist)

        return hist.flatten()

    def _conv(self, img):
        edges = self.edge_detector(img)
        edges_flat = edges.ravel()
        hist, bins = np.histogram(edges_flat, bins=self.bin, range=(0, 255))

        if self.normalize:
            hist = hist / np.sum(hist)

        return hist

    def _get_edge_detector(self, edge_detector):
        if edge_detector == "canny":
            return lambda img: cv2.Canny(img, 100, 200)
        elif edge_detector == "sobel":
            def sobel(image, ksize=3):
                x_edges = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
                y_edges = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
                return np.sqrt(x_edges**2 + y_edges**2)
            return lambda img: sobel(img, self.ksize)
        elif edge_detector == "prewitt":
            raise NotImplementedError("This detector is not implemented yet")
        else:
            raise ValueError(f"Invalid edge detector: {edge_detector}")
