import numpy as np

from .mytypes import *


def regularization_l2(w: ndarray, l2_lambda: float) -> float:

    return l2_lambda / 2 * np.sum(w ** 2).item()
