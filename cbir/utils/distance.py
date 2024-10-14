from __future__ import print_function

from typing import Literal

import numpy as np
from scipy import spatial

d2s_typing = Literal["exp", "log", "logistic", "gaussian", "inverse"]
def get_d2s_transform(
    distance_transform: d2s_typing,
    **kwargs,
) -> callable:
    """
    Returns a callable that transforms the given distance into a score.
    
    Parameters
    ----------
    distance_transform : str
        The type of distance transform to use. One of 'exp', 'log', 'logistic', 'gaussian', 'inverse'.
    **kwargs : dict
        Additional keyword arguments to pass to the chosen transform. For example, the 'gaussian' transform
        requires a 'sigma' parameter.
    
    Returns
    -------
    callable
        A function that takes a distance and returns a transformed score.
    """
    if distance_transform == "exp":
        return lambda x: np.exp(-x)
    elif distance_transform == "log":
        return lambda x: -np.log(x)
    elif distance_transform == "logistic":
        return lambda x: 1 / (1 + np.exp(-x))
    elif distance_transform == "gaussian":
        sigma = kwargs.get("sigma", 1)
        return lambda x: np.exp(-(x**2) / (2 * sigma**2))
    elif distance_transform == "inverse":
        return lambda x: 1 / 1 + x
    else:
        raise ValueError(f"Invalid distance transform: {distance_transform}")


def distance(
    v1, v2, d_type: Literal["absolute", "cosine", "square", "d2-norm"] = "cosine"
):
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
