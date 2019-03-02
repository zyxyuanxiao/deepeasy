import numpy as np

from .types import *


def inverted_dropout(a: ndarray,
                     keep_prob: float,
                     seed: Optional[int] = None) -> ndarray:
    np.random.seed(seed)
    d = np.random.rand(*a.shape) < keep_prob
    return a * d / keep_prob
