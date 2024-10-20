from __future__ import annotations

import itertools
import os
from math import sqrt
from typing import Literal

import cv2
import numpy as np

from cbir.feature_extractor import (BatchFeatureExtractor, FeatureExtractor,
                                    SingleFeatureExtractor)


class FourierDescriptor(SingleFeatureExtractor):
    def __init__(
        self,
        n_slice=10,
        h_type="region",
        num_coeffs=10,
        normalize=True,
        **kwargs,
    ) -> None:
        self.n_sclice = n_slice
        self.num_coeffs = num_coeffs
        self.h_type = h_type
        self.normalize = normalize

    def __call__(self, input: np.ndarray | os.PathLike):
        img = super().read_image(input)

        height, width, channel = img.shape

        if self.h_type == "global":
            hist = self.__fourier_descriptor(img)

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
                    hist[hs][ws] = self.__fourier_descriptor(img_r)

        if self.normalize:
            hist /= np.sum(hist)

        return hist.flatten()

    def __fourier_descriptor(self, img):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200) 

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Select a contour
        cnt = contours[0]

        # Resample the contour to a fixed number of points
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Compute Fourier descriptors
        fd = cv2.dft(approx.astype(np.float32))

        # Extract magnitude and phase
        magnitude = cv2.magnitude(fd[:, 0], fd[:, 1])
        phase = cv2.phase(fd[:, 0], fd[:, 1])

        # Select a subset of Fourier coefficients
        fd_selected = magnitude[:self.num_coeffs]
        
        return fd_selected
