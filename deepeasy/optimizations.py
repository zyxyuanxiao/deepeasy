""""神经网络中的梯度下降优化算法。

create:   2019-02-26
modified: 2019-02-26
"""

import numpy as np

from .types import *
from .env import *


class GD:

    def __init__(self, lr=0.01) -> None:
        self.lr: float = lr

    def update(self, params, backward_caches) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class SGD(GD):

    def __init__(self, lr=0.01) -> None:
        super().__init__(lr)

    def update(self, params, backward_caches) -> None:
        for key in params:
            params[key] -= self.lr * backward_caches[key]


class Momentum(GD):

    def __init__(self, lr=0.01, beta=0.9) -> None:
        super().__init__(lr)
        self.beta = beta
        self.v = {}

    def update(self, params, backward_caches) -> None:

        for k in params:
            self.v[k] = self.beta * self.v.get(k, 0)\
                        + (1. - self.beta) * backward_caches[k]
            params[k] -= self.lr * self.v[k]

    def reset(self) -> None:
        self.v.clear()


class RMSprop(GD):

    def __init__(self, lr=0.01, beta=0.999) -> None:
        super().__init__(lr)
        self.beta = beta
        self.s = {}

    def update(self, params, backward_caches) -> None:

        for k in params:
            self.s[k] = self.beta * self.s.get(k, 0)\
                        + (1. - self.beta) * backward_caches[k]**2
            params[k] -= self.lr * (backward_caches[k] / np.sqrt(self.s[k] + DELTA))

    def reset(self) -> None:
        self.s.clear()


class Adam(GD):

    def __init__(self, lr=0.01, beta1=0.9, beta2=0.999) -> None:
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.v = {}
        self.s = {}
        self.iter = 0

    def update(self, params, backward_caches) -> None:

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter)\
            / (1.0 - self.beta1**self.iter)

        for k in params:
            self.v[k] = self.beta1 * self.v.get(k, 0)\
                + (1. - self.beta1) * backward_caches[k]

            self.s[k] = self.beta2 * self.s.get(k, 0) \
                + (1. - self.beta2) * backward_caches[k]**2

            params[k] -= lr_t * (self.v[k] / np.sqrt(self.s[k] + DELTA))

    def reset(self) -> None:
        self.v.clear()
        self.s.clear()


def get_gd(name: str,
           lr: float,
           beta1: float,
           beta2: float) -> GD:

    name = name.lower()
    if name == 'sgd':
        return SGD(lr)
    elif name == 'momentum':
        return Momentum(lr, beta1)
    elif name == 'rmsprop':
        return RMSprop(lr, beta1)
    elif name == 'adam':
        return Adam(lr, beta1, beta2)
    else:
        raise Exception('Non-supported optimization algorithm')
