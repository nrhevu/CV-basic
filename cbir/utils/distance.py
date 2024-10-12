from __future__ import print_function

from typing import Literal

import numpy as np
from scipy import spatial


def distance(
    v1, v2, d_type: Literal["absolute", "cosine", "square", "d2-norm"] = "cosine"
):
    """
    Calculate distance between two vectors

    Parameters
    ----------
    v1 : array
        vector 1
    v2 : array
        vector 2
    d_type : str, optional
        type of distance to calculate, by default "cosine"

        - "absolute": absolute difference
        - "cosine": cosine distance
        - "square": square difference
        - "d2-norm": D2 norm

    Returns
    -------
    float
        distance between two vectors
    """
    assert v1.shape == v2.shape, "shape of two vectors need to be same!"

    if d_type == "absolute":
        return np.sum(np.absolute(v1 - v2))
    elif d_type == "d2-norm":
        return 2 - 2 * np.dot(v1, v2)
    elif d_type == "cosine":
        return spatial.distance.cosine(v1, v2)
    elif d_type == "square":
        return np.sum((v1 - v2) ** 2)
    else:
        raise ValueError(f"Invalid distance type: {d_type}")