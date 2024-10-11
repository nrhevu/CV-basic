from os import PathLike
from numpy import ndarray
from cbir.pipeline.cbir import CBIR, ImageSearchObject

class RGBHistogramCBRI(CBIR):
    def fit(self, images: list) -> None:
        return super().fit(images)
    
    def predict(self, image: ndarray | PathLike) -> list[ImageSearchObject]:
        return super().predict(image)