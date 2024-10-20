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
        stride=(1, 1),
        n_slice=10,
        h_type="region",
        normalize=True,
    ) -> None:
        self.stride = stride
        self.n_slice = n_slice
        self.h_type = h_type
        self.normalize = normalize
        self.edge_kernels = np.array(
            [
                [
                    # vertical
                    [1, -1],
                    [1, -1],
                ],
                [
                    # horizontal
                    [1, 1],
                    [-1, -1],
                ],
                [
                    # 45 diagonal
                    [sqrt(2), 0],
                    [0, -sqrt(2)],
                ],
                [
                    # 135 diagnol
                    [0, sqrt(2)],
                    [-sqrt(2), 0],
                ],
                [
                    # non-directional
                    [2, -2],
                    [-2, 2],
                ],
            ]
        )

    def __call__(self, input: np.ndarray | os.PathLike):
        img = super().read_image(input)
        
        height, width, channel = img.shape

        if self.h_type == "global":
            hist = self._conv(img, stride=self.stride, kernels=self.edge_kernels)

        elif self.h_type == "region":
            hist = np.zeros((self.n_slice, self.n_slice, self.edge_kernels.shape[0]))
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
                    hist[hs][ws] = self._conv(
                        img_r, stride=self.stride, kernels=self.edge_kernels
                    )

        if self.normalize:
            hist /= np.sum(hist)

        return hist.flatten()

    def _conv(self, img, stride, kernels, normalize=True):
        H, W, C = img.shape
        conv_kernels = np.expand_dims(kernels, axis=3)
        conv_kernels = np.tile(conv_kernels, (1, 1, 1, C))
        assert list(conv_kernels.shape) == list(kernels.shape) + [
            C
        ]  # check kernels size

        sh, sw = stride
        kn, kh, kw, kc = conv_kernels.shape

        hh = int((H - kh) / sh + 1)
        ww = int((W - kw) / sw + 1)

        hist = np.zeros(kn)

        for idx, k in enumerate(conv_kernels):
            for h in range(hh):
                hs = int(h * sh)
                he = int(h * sh + kh)
                for w in range(ww):
                    ws = w * sw
                    we = w * sw + kw
                    hist[idx] += np.sum(img[hs:he, ws:we] * k)  # element-wise product

        if normalize:
            hist /= np.sum(hist)

        return hist
