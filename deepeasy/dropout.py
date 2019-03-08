"""

create:   2019-03-08
modified:
"""

import numpy as np

from .log import logger
from .mytypes import *


def inverted_dropout(a: ndarray,
                     keep_prob: float,
                     seed: Optional[int] = None) -> ndarray:
    np.random.seed(seed)
    d = np.random.rand(*a.shape) < keep_prob
    return a * d / keep_prob


def update_keep_prob(layers: List, keep_probs: Tuple[float]) -> None:
    """更新每层的 keep_prob。"""

    if len(keep_probs) != len(layers):
        logger.error('dropout_keep_probs 长度必须与神经网络层数一致。')
        exit(1)

    for layer, dropout_keep_prob in zip(layers, keep_probs):
        check_keep_prob_range(dropout_keep_prob)
        layer.dropout_keep_prob = dropout_keep_prob


def check_keep_prob_range(keep_prob: float) -> None:
    if keep_prob < 0. or keep_prob > 1.:
        logger.error('dropout_keep_prob 的值必须在 0 到 1 之间！')
        exit(1)
