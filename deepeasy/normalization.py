import numpy as np

from .mytypes import *
from .env import *


def whitening(z: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
    z_mean = z.mean(axis=SAMPLE_AXIS, keepdims=True)
    x = z - z_mean
    z_var = x.var(axis=SAMPLE_AXIS, keepdims=True)
    return x / np.sqrt(z_var + EPSILON), z_mean, z_var
