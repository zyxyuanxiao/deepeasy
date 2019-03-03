import numpy as np

from .types import *
from .env import *


def get_accuracy(a: ndarray, y: ndarray) -> float:
    feature_num = y.shape[LABEL_FEATURES_NUM_AXIS]
    batch_size = y.shape[LABELS_NUM_AXIS]

    if feature_num == 1:
        a = np.where(a > 0.5, 1, 0)
        return np.sum(a == y) / batch_size
    else:
        return np.sum(
            a.argmax(axis=LABEL_FEATURES_NUM_AXIS)
            == y.argmax(axis=LABEL_FEATURES_NUM_AXIS)
        ) / batch_size
