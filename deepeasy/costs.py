"""机器学习中的 cost 函数。

create:   2019-02-23
modified:
"""

import numpy as np

from .types import *
from .env import *


def cost_mean_squared(a: ndarray, y: ndarray) -> float:
    samples_num = y.shape[1]
    return 0.5 / samples_num * np.sum((a-y)**2)


def cost_cross_entropy(a: ndarray, y: ndarray) -> float:
    feature_num = a.shape[LABEL_FEATURES_NUM_AXIS]
    batch_size = a.shape[LABELS_NUM_AXIS]

    if feature_num == 1:
        cost = -np.sum(y * np.log(a + DELTA) + (1 - y) * np.log(1 - a)) / batch_size
    else:
        cost = -np.sum(y * np.log(a + DELTA)) / batch_size
    return cost
