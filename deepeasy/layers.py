import numpy as np

from .activations import get_activation_func
from .dropout import inverted_dropout
from .types import *
from .env import *


class Layer:
    """神经网络中的每一层。"""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: Optional[str],
                 dropout_keep_prob: float = 1.) -> None:

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation.lower() if activation else ''
        self.dropout_keep_prob = dropout_keep_prob

        self.params: Dict[str, ndarray] = {}

        self.init_params()

        self.g, self.g_backward = get_activation_func(activation)

        # 缓存
        self.forward_caches: Dict[str, ndarray] = {}
        self.backward_caches: Dict[str, ndarray] = {}

    def forward(self, a_pre: ndarray) -> ndarray:

        z = a_pre @ self.params['w'] + self.params['b']
        a = self.g(z)  # shape = (该层神经元数, 样本数)

        # dropout
        if self.dropout_keep_prob < 1:
            a = inverted_dropout(a, self.dropout_keep_prob)

        self.forward_caches['a_pre'] = a_pre
        self.forward_caches['z'] = z
        self.forward_caches['a'] = a

        return a

    def backward(self, da: Optional[ndarray], y: ndarray) -> ndarray:
        if da is None:
            dz = (self.forward_caches['a'] - y) / y.shape[LABELS_NUM_AXIS]
        else:
            dz = da * self.g_backward(self.forward_caches['z'])

        dw = self.forward_caches['a_pre'].T @ dz

        db = np.sum(dz, axis=LABELS_NUM_AXIS, keepdims=True)

        self.backward_caches['w'] = dw
        self.backward_caches['b'] = db

        da = dz @ self.params['w'].T
        return da

    def init_params(self) -> None:

        # Xavier Initializer
        # 除以每个神经元被连接数的平方根
        # 如果用 ReLU，除以每个神经元被连接数的平方根乘 2，会更好
        n = self.input_dim
        if self.activation.lower() == 'relu':
            n /= 2
        self.params['w'] = np.random.randn(
            self.input_dim, self.output_dim
        ) / np.sqrt(n)

        self.params['b'] = np.zeros(
            shape=(1, self.output_dim)
        )

        # self.params['gamma'] = np.ones(
        #     shape=(self.input_dim, self.output_dim)
        # )
        #
        # self.params['beta'] = np.zeros(
        #     shape=(1, self.output_dim)
        # )
