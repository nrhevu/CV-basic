from __future__ import annotations

from os import PathLike
from typing import Literal

import cv2
import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm

from cbir.feature_extractor import SingleFeatureExtractor


class SIFTBOWExtractor(SingleFeatureExtractor):
    def __init__(self, mode : Literal["bow", "tfidf"] = "tfidf") -> None:
        super().__init__()
        self.feat = []
        self.mode = mode
    
    def __call__(self, input: ndarray | PathLike):
        img = super().read_image(input)  
        # Return vector if mode is bow
        if self.mode == "bow":
            bow_hist = self.__bow(img)
            return bow_hist / max(bow_hist)
        elif self.mode == "tfidf":
            tfidf = self.__tfidf(img)
            return tfidf
        
    
    def fit(self, images: list, k=200):
        for img in tqdm(images, desc="Extracting Features"):
            feature = self.__computeFeatures(img)
            if feature is not None:
                self.feat.append(feature)
            
        # Stack all features together
        alldes = np.vstack(self.feat)
        
        print("Fit Kmeans clustering to create BOW")
        # Perform Kmeans clustering and get the cluster centers to reduce dimensionality
        alldes = np.float32(alldes) 
        self.kmeans = KMeans(n_clusters=k, random_state=0).fit(alldes)  
        codebook, distortion = self.kmeans.cluster_centers_, self.kmeans.inertia_
        
        print("Fit IDF for TF-IDF Transformation")
        # Fit IDF Transformation for TF-IDF 
        # Create Bag-of-word list
        bow = []

        # Get label for each image, and put into a histogram (BoW)
        for f in self.feat:
            code = self.kmeans.predict(f)
            bow_hist, _ = np.histogram(code, self.kmeans.n_clusters)
            bow.append(bow_hist)
            
        self.tfidf_transformer = TfidfTransformer(smooth_idf=True)
        self.tfidf_transformer.fit(bow)
        
        print("Complete Fitting SIFT BOW Extractor")
    
    def __bow(self, img: np.ndarray):
        feature = self.__computeFeatures(img)
        # Get label for each image, and put into a histogram (BoW)
        try:
            code = self.kmeans.predict(feature)
        except:
            raise Exception("Must call fit() to create codebook before feature extraction")
        bow_hist, _ = np.histogram(code, self.kmeans.n_clusters)
        
        return bow_hist
    
    def __tfidf(self, img: np.ndarray):
        bow_hist = self.__bow(img)
        return self.tfidf_transformer.transform([bow_hist]).toarray()[0]
    
    def clean(self):
        self.feat = []

    def __computeFeatures(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        sift = cv2.SIFT_create()
        kps, des = sift.detectAndCompute(img, None)
        return des