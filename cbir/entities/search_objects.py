import numpy as np


class ImageSearchObject():
    def __init__(self, image: np.ndarray, score: float) -> None:
        self.image = image
        self.score = score
