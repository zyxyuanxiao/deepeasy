"""神经网络中的激活函数。

create:   2019-02-18
modified: 2019-02-24
"""

import numpy as np

from .types import *
from .env import *


def relu(z: ndarray) -> ndarray:
    return np.maximum(z, 0)


def relu_prime(z: ndarray) -> ndarray:
    return z > 0


def softmax(z: ndarray) -> ndarray:
    c = np.max(z, axis=LABEL_FEATURES_NUM_AXIS, keepdims=True)
    exp_z = np.exp(z-c)  # 防溢出
    sum_exp_z = np.sum(exp_z, axis=LABEL_FEATURES_NUM_AXIS, keepdims=True)
    return exp_z / sum_exp_z


def softmax_prime(z: ndarray) -> ndarray:
    return softmax(z) * (1 - softmax(z))


def tanh(z: ndarray) -> ndarray:
    return np.tanh(z)


def tanh_prime(z: ndarray) -> ndarray:
    return 1 - tanh(z) * tanh(z)


def sigmoid(z: ndarray) -> ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z: ndarray) -> ndarray:
    return sigmoid(z) * (1 - sigmoid(z))


def get_activation_func(name: Optional[str] = None) -> Tuple[Callable, Callable]:
    if name is None:  # 代表不使用激活函数
        return lambda z: z, lambda da, z: da

    name = name.lower()
    if name == 'relu':
        return relu, relu_prime
    elif name == 'softmax':
        return softmax, softmax_prime
    elif name == 'tanh':
        return tanh, tanh_prime
    elif name == 'sigmoid':
        return sigmoid, sigmoid_prime
    else:
        raise Exception('Non-supported activation function')
