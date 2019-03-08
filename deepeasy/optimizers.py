""""神经网络中的梯度下降优化算法。

create:   2019-02-26
modified: 2019-02-26
"""

import numpy as np

from .layers import Layer
from .mytypes import *


class GD:

    def __init__(self, lr: float = 0.01) -> None:
        self.lr: float = lr

    def update(self, layers: List[Layer]) -> None:

        raise NotImplementedError

    def reset(self) -> None:
        pass


class SGD(GD):

    def __init__(self, lr: float = 0.01) -> None:
        super().__init__(lr)

    def update(self, layers: List[Layer]) -> None:

        for layer in layers:
            for theta in layer.params:
                layer.params[theta] -= self.lr * layer.backward_caches[theta]


class Momentum(GD):

    def __init__(self, lr: float = 0.01, beta: float = 0.9) -> None:
        super().__init__(lr)
        self.beta = beta
        self.v = {}

    def update(self, layers: List[Layer]) -> None:

        for i, layer in enumerate(layers, 1):
            for theta in layer.params:
                k = f'{theta}{i}'
                self.v[k] = self.beta * self.v.get(k, 0.) \
                            + self.lr * layer.backward_caches[theta]
                layer.params[theta] -= self.v[k]

    def reset(self) -> None:
        self.v.clear()


class RMSprop(GD):

    def __init__(self, lr: float = 0.001, beta: float = 0.9) -> None:
        super().__init__(lr)
        self.beta = beta
        self.s = {}

    def update(self, layers: List[Layer]) -> None:

        for i, layer in enumerate(layers, 1):
            for theta in layer.params:
                k = f'{theta}{i}'
                self.s[k] = self.beta * self.s.get(k, 0.) \
                            + (1. - self.beta) * layer.backward_caches[theta]**2
                layer.params[theta] -= \
                    self.lr * (layer.backward_caches[theta]
                               / np.sqrt(self.s[k] + 1e-6))

    def reset(self) -> None:
        self.s.clear()


class Adam(GD):

    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999) -> None:
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.v = {}
        self.s = {}
        self.iter = 0

    def update(self, layers: List[Layer]) -> None:

        self.iter += 1
        self.iter %= 170
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter)\
            / (1.0 - self.beta1**self.iter)

        for i, layer in enumerate(layers, 1):
            for theta in layer.params:
                k = f'{theta}{i}'
                self.v[k] = self.beta1 * self.v.get(k, 0.)\
                    + (1. - self.beta1) * layer.backward_caches[theta]

                self.s[k] = self.beta2 * self.s.get(k, 0.) \
                    + (1. - self.beta2) * layer.backward_caches[theta]**2

                layer.params[theta] -= \
                    lr_t * (self.v[k] / (np.sqrt(self.s[k]) + 1e-8))

    def reset(self) -> None:
        self.iter = 0
        self.v.clear()
        self.s.clear()


class Nadam(Adam):

    def __init__(self, lr: float = 0.002, beta1: float = 0.9, beta2: float = 0.999) -> None:
        super().__init__(lr, beta1, beta2)

    def update(self, layers: List[Layer]) -> None:

        self.iter += 1
        self.iter %= 170
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter)\
            / (1.0 - self.beta1**self.iter)

        for i, layer in enumerate(layers, 1):
            for theta in layer.params:
                k = f'{theta}{i}'
                self.v[k] = self.beta1 * self.v.get(k, 0.)\
                    + (1. - self.beta1) * layer.backward_caches[theta]

                self.s[k] = self.beta2 * self.s.get(k, 0.) \
                    + (1. - self.beta2) * layer.backward_caches[theta]**2

                layer.params[theta] -= lr_t * ((self.beta1 * self.v[k] + (1. - self.beta1) * layer.backward_caches[theta]) / (np.sqrt(self.s[k]) + 1e-7))


def get_optimizer(name: str,
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
    elif name == 'nadam':
        return Nadam(lr, beta1, beta2)
    else:
        raise Exception('Non-supported optimization algorithm')
