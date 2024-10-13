from __future__ import annotations

import itertools
import os

import cv2
import numpy as np

from cbir.feature_extractor import (BatchFeatureExtractor, FeatureExtractor,
                                    SingleFeatureExtractor)


class RGBHistogram(SingleFeatureExtractor):
    def __init__(
        self,
        n_bin=12,  # histogram bins
        n_slice=3,  # slice image
        h_type="region",  # global or region) -> None
        normalize=True,
    ) -> None:
        self.n_bin = n_bin
        self.n_slice = n_slice
        self.h_type = h_type
        self.normalize = normalize

    def __call__(self, input: np.ndarray | os.PathLike):
        """
        Calculate a histogram for given image.

        Parameters
        ----------
        input : np.ndarray | os.PathLike
            The input image.

        Returns
        -------
        hist : list
            A list of histogram features for the input image.
        """
        img = super().read_image(input)

        height, width, channel = img.shape

        if self.h_type == "global":
            hist = self._count_hist(img, self.n_bin)

        elif self.h_type == "region":
            hist = np.zeros((self.n_slice, self.n_slice, self.n_bin**channel))
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
                    hist[hs][ws] = self._count_hist(img_r, self.n_bin)

        if self.normalize:
            hist /= np.sum(hist)

        return hist.flatten()

    def _count_hist(self, input, n_bin, channel=3):
        color_histogram = cv2.calcHist(
            [input],
            np.arange(channel),
            None,
            [n_bin, n_bin, n_bin],
            [0, 256, 0, 256, 0, 256],
        )

        return color_histogram.flatten()
