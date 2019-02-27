""""神经网络中的梯度下降优化算法。

create:   2019-02-26
modified: 2019-02-26
"""

import numpy as np

from .types import *
from .env import *


def get_gd_func(name: str) -> Callable:

    name = name.lower()
    if name == 'sgd':
        return gd_sgd
    elif name == 'momentum':
        return gd_momentum
    elif name == 'rmsprop':
        return gd_rmsprop
    elif name == 'adam':
        return gd_adam
    else:
        raise Exception('Non-supported optimization algorithm')


def gd_sgd(backward_caches: Dict,
           optimization_caches: Dict,
           beta1: float,
           beta2: float,
           t: int) -> Dict:

    return backward_caches


def gd_momentum(backward_caches: Dict,
                optimization_caches: Dict,
                beta1: float,
                beta2: float,
                t: int) -> Dict:

    layers_count = _get_layer_count(backward_caches)

    d = {}
    for layer_idx in range(1, layers_count + 1):
        optimization_caches[f'vdw{layer_idx}'] = \
            beta1 * optimization_caches.get(f'vdw{layer_idx}', 0) \
            + (1 - beta1) * backward_caches[f'dw{layer_idx}']
        optimization_caches[f'vdb{layer_idx}'] = \
            beta1 * optimization_caches.get(f'vdb{layer_idx}', 0) \
            + (1 - beta1) * backward_caches[f'db{layer_idx}']

        d[f'dw{layer_idx}'] = optimization_caches[f'vdw{layer_idx}']
        d[f'db{layer_idx}'] = optimization_caches[f'vdb{layer_idx}']

    return d


def gd_rmsprop(backward_caches: Dict,
               optimization_caches: Dict,
               beta1: float,
               beta2: float,
               t: int) -> Dict:

    layers_count = _get_layer_count(backward_caches)

    d = {}
    for layer_idx in range(1, layers_count + 1):
        optimization_caches[f'sdw{layer_idx}'] = \
            beta2 * optimization_caches.get(f'sdw{layer_idx}', 0) \
            + (1 - beta2) * backward_caches[f'dw{layer_idx}']**2
        optimization_caches[f'sdb{layer_idx}'] = \
            beta2 * optimization_caches.get(f'sdb{layer_idx}', 0) \
            + (1 - beta2) * backward_caches[f'db{layer_idx}']**2

        d[f'dw{layer_idx}'] = backward_caches[f'dw{layer_idx}'] \
            / np.sqrt(optimization_caches[f'sdw{layer_idx}'] + DELTA)
        d[f'db{layer_idx}'] = backward_caches[f'db{layer_idx}'] \
            / np.sqrt(optimization_caches[f'sdb{layer_idx}'] + DELTA)

    return d


def gd_adam(backward_caches: Dict,
            optimization_caches: Dict,
            beta1: float,
            beta2: float,
            t: int) -> Dict:

    layers_count = _get_layer_count(backward_caches)

    gd_momentum(backward_caches, optimization_caches, beta1, beta2, t)
    gd_rmsprop(backward_caches, optimization_caches, beta1, beta2, t)
    _corret_gd(backward_caches, optimization_caches, beta1, beta2, t)

    for layer_idx in range(1, layers_count + 1):
        backward_caches[f'dw{layer_idx}'] = \
            optimization_caches[f'vdw{layer_idx}'] \
            / np.sqrt(optimization_caches[f'sdw{layer_idx}'] + DELTA)
        backward_caches[f'db{layer_idx}'] = \
            optimization_caches[f'vdb{layer_idx}'] \
            / np.sqrt(optimization_caches[f'sdb{layer_idx}'] + DELTA)

    return backward_caches


def _corret_gd(backward_caches: Dict,
               optimization_caches: Dict,
               beta1: float,
               beta2: float,
               t: int) -> None:
    """Adam 中需要修正 Momentum 和 RMSprop，
    单独使用 Momentum 或 RMSprop 则不需要。"""

    t = t % 170  # 防 beta**t 溢出，实际已经够接近 0 了
    layers_count = _get_layer_count(backward_caches)

    for layer_idx in range(1, layers_count + 1):
        optimization_caches[f'vdw{layer_idx}'] = \
            optimization_caches[f'vdw{layer_idx}'] / (1 - beta1**t)
        optimization_caches[f'vdb{layer_idx}'] = \
            optimization_caches[f'vdb{layer_idx}'] / (1 - beta1**t)
        optimization_caches[f'sdw{layer_idx}'] = \
            optimization_caches[f'sdw{layer_idx}'] / (1 - beta2**t)
        optimization_caches[f'sdb{layer_idx}'] = \
            optimization_caches[f'sdb{layer_idx}'] / (1 - beta2**t)


def _get_layer_count(backward_caches: Dict) -> int:
    return len(backward_caches) // 2
