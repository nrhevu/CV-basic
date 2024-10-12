from os import PathLike
from numpy import ndarray
from cbir.cbir import CBIR, ImageSearchObject

class RGBHistogramCBRI(CBIR):
    def __init__(self) -> None:
        super().__init__()
    
    def indexing(self, images: list) -> None:
        return super().fit(images)
    
    def retrieve(self, image: ndarray | PathLike) -> list[ImageSearchObject]:
        return super().predict(image)