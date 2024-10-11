from __future__ import annotations

import itertools
import os

import cv2
import numpy as np

from cbir.featuring.feature_extractor import FeatureExtractor


class RGBHistogram(FeatureExtractor):
    def __init__(
        self,
        n_bin=12,  # histogram bins
        n_slice=3,  # slice image
        h_type="region",  # global or region) -> None
        normalize=True 
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
        bins = np.linspace(0, 256, self.n_bin+1, endpoint=True)  # slice bins equally for each channel
    
        if self.h_type == 'global':
            hist = self._count_hist(img, self.n_bin, bins, channel)
    
        elif self.h_type == 'region':
            hist = np.zeros((self.n_slice, self.n_slice, self.n_bin ** channel))
            h_silce = np.around(np.linspace(0, height, self.n_slice+1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(0, width, self.n_slice+1, endpoint=True)).astype(int)
            for hs in range(len(h_silce)-1):
                for ws in range(len(w_slice)-1):
                    img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
                    hist[hs][ws] = self._count_hist(img_r, self.n_bin, bins, channel)
                    
    
        if self.normalize:
            hist /= np.sum(hist)
    
        return hist.flatten()
    
    
    def _count_hist(self, input, n_bin, bins, channel):
        """
        Count the histogram of the input image given the bins and channel.

        Parameters
        ----------
        input : np.ndarray
            The input image.
        n_bin : int
            The number of bins for each channel.
        bins : np.ndarray
            The bins for each channel.
        channel : int
            The number of color channels.

        Returns
        -------
        hist : np.ndarray
            The histogram of the input image.
        """
        img = input.copy()
        bins_idx = {key: idx for idx, key in enumerate(itertools.product(np.arange(n_bin), repeat=channel))}  # permutation of bins
        hist = np.zeros(n_bin ** channel)
    
        # cluster every pixels
        for idx in range(len(bins)-1):
            img[(input >= bins[idx]) & (input < bins[idx+1])] = idx
        # add pixels into bins
        height, width, _ = img.shape
        for h in range(height):
            for w in range(width):
                b_idx = bins_idx[tuple(img[h,w])]
                hist[b_idx] += 1
    
        return hist
