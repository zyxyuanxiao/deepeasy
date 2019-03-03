from .types import *
from .env import *


def whitening(z: ndarray) -> Tuple[ndarray, float, float]:
    mu = z.mean(axis=LABELS_NUM_AXIS, keepdims=True)
    x = z - mu
    sigma = x.std(axis=LABELS_NUM_AXIS, keepdims=True)
    return x / (sigma + EPSILON), mu, sigma
