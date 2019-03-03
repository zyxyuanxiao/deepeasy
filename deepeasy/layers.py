import numpy as np

from .activations import get_activation_func
from .dropout import inverted_dropout
from .normalization import whitening
from .types import *
from .env import *


class Layer:
    """神经网络中的每一层。"""

    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 activation: Optional[str],
                 layer_idx: int,
                 *,
                 batch_normalization: bool,
                 dropout_keep_prob: float) -> None:

        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.activation: str = activation.lower() if activation else ''
        self.layer_idx: int = layer_idx
        self.is_output_layer: bool = False
        self.batch_normalization: bool = batch_normalization
        self.dropout_keep_prob: float = dropout_keep_prob

        self.params: Dict[str, ndarray] = {}

        self.init_params()

        self.g, self.g_prime = get_activation_func(activation)

        # 缓存
        self.forward_caches: Dict[str, ndarray] = {}
        self.backward_caches: Dict[str, ndarray] = {}

    def forward(self, a_pre: ndarray) -> ndarray:

        z = a_pre @ self.params['w']
        if self.batch_normalization and not self.is_output_layer:
            z_white, mu, sigma = whitening(z)
            self.forward_caches['z_tilde'] = z_white
            z_tilde = z_white @ self.params['gamma'] + self.params['beta']
            self.forward_caches['z_tilde'] = z_tilde
        else:
            z += self.params['b']

        a = self.g(z)  # shape = (该层神经元数, 样本数)

        # dropout
        if self.dropout_keep_prob < 1:
            a = inverted_dropout(a, self.dropout_keep_prob)

        self.forward_caches['a_pre'] = a_pre
        self.forward_caches['z'] = z
        self.forward_caches['a'] = a

        return a

    def backward(self, da: Optional[ndarray], y: ndarray) -> ndarray:
        if self.is_output_layer:
            dz = (self.forward_caches['a'] - y) / y.shape[LABELS_NUM_AXIS]
        else:
            if self.batch_normalization:
                dz_tilde = da * self.g_prime(self.forward_caches['z_tilde'])
                dgamma = self.forward_caches['z_white'].T @ dz_tilde
                self.backward_caches['dgamma'] = dgamma
                dbeta = np.sum(dz_tilde, axis=LABELS_NUM_AXIS, keepdims=True)
                self.backward_caches['dbeta'] = dbeta
                dz_white = dz_tilde @ self.params['gamma'].T
                # TODO
                dz = 1
            else:
                dz = da * self.g_prime(self.forward_caches['z'])

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

        if self.batch_normalization and not self.is_output_layer:
            self.params['gamma'] = np.ones(
                shape=(self.input_dim, self.output_dim)
            )

            self.params['beta'] = np.zeros(
                shape=(1, self.output_dim)
            )
        else:
            self.params['b'] = np.zeros(
                shape=(1, self.output_dim)
            )

    def reset(self) -> None:
        self.init_params()
        self.forward_caches.clear()
        self.backward_caches.clear()
