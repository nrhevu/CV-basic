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
        self.n_slice = n_slice
        self.num_coeffs = num_coeffs
        self.h_type = h_type
        self.normalize = normalize

    def __call__(self, input: np.ndarray | os.PathLike):
        img = super().read_image(input)

        height, width, channel = img.shape

        if self.h_type == "global":
            hist = self.__fourier_descriptor(img)

        elif self.h_type == "region":
            hist = np.zeros((self.n_slice, self.n_slice, self.num_coeffs))
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

        # Apply Canny
        edges = cv2.Canny(gray, 100, 200)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get the largest contour (assuming it is the main shape)
        try:
            contour = max(contours, key=cv2.contourArea)
        except:
            return np.zeros(self.num_coeffs)

        # Convert contour to complex numbers
        contour_complex = np.empty(contour.shape[0], dtype=complex)
        contour_complex.real = contour[:, 0, 0]  # x coordinates
        contour_complex.imag = contour[:, 0, 1]  # y coordinates
        
        # Perform Discrete Fourier Transform (DFT)
        fourier_result = np.fft.fft(contour_complex)
        
        # # Normalize the Fourier descriptors to achieve scale invariance
        # fourier_result /= np.abs(fourier_result[1])
        
        # Retain only the first `self.num_coeffs` descriptors for comparison
        if self.num_coeffs is not None:
            descriptors = fourier_result[:self.num_coeffs]
        else:
            descriptors = fourier_result

        if len(descriptors) < self.num_coeffs:
            descriptors = np.pad(np.abs(descriptors), (0, self.num_coeffs - len(descriptors)))
            return descriptors
            
        return np.abs(descriptors)