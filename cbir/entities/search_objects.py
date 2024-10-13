import numpy as np


class ImageSearchObject():
    def __init__(self, index: int, score: float, image = None) -> None:
        self.index = index
        self.score = score
        self.image = image