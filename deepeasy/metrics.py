import numpy as np

from .mytypes import *
from .env import *


def get_accuracy(a: ndarray, y: ndarray) -> float:
    feature_num = y.shape[FEATURE_AXIS]
    batch_size = y.shape[SAMPLE_AXIS]

    if feature_num == 1:
        a = np.where(a > 0.5, 1, 0)
        return np.sum(a == y) / batch_size
    else:
        return np.sum(
            a.argmax(axis=FEATURE_AXIS)
            == y.argmax(axis=FEATURE_AXIS)
        ) / batch_size
