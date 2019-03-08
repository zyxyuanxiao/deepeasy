"""

create:   2019-03-08
modified:
"""

import numpy as np

from .mytypes import *
from .env import *


def whitening_train(z: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    z_mean = z.mean(axis=SAMPLE_AXIS, keepdims=True)
    x = z - z_mean
    z_var = x.var(axis=SAMPLE_AXIS, keepdims=True)
    return x / np.sqrt(z_var + EPSILON), z_mean, z_var


def whitening_predict(x: ndarray, mu: ndarray, sigma_square: ndarray) -> ndarray:

    return (x - mu) / np.sqrt(sigma_square + EPSILON)
