import numpy as np

from .activations import get_activation_func
from .dropout import inverted_dropout
from .normalization import whitening
from .mytypes import *
from .env import *


class Layer:
    """神经网络中的每一层。"""

    def __init__(self,
                 neural_network,
                 input_dim: int,
                 output_dim: int,
                 activation: Optional[str],
                 layer_idx: int,
                 *,
                 is_output_layer: bool = False,
                 dropout_keep_prob: float) -> None:

        self.neural_network = neural_network
        self.input_dim: int = input_dim
        self.output_dim: int = output_dim
        self.activation: str = activation.lower() if activation else ''
        self.layer_idx: int = layer_idx
        self.is_output_layer: bool = is_output_layer
        self.dropout_keep_prob: float = dropout_keep_prob

        self.params: Dict[str, ndarray] = {}
        self.init_params()

        self.g, self.g_prime = get_activation_func(activation)

        # 缓存
        self.forward_caches: Dict[str, ndarray] = {}
        self.backward_caches: Dict[str, ndarray] = {}

    def forward(self, a_pre: ndarray) -> ndarray:

        z = a_pre @ self.params['w']
        if self.neural_network.batch_normalization and not self.is_output_layer:
            z_white, mu, sigma_square = whitening(z)
            self.forward_caches['z_white'] = z_white
            self.forward_caches['mu'] = mu
            self.forward_caches['sigma_square'] = sigma_square
            z_tilde = z_white * self.params['gamma'] + self.params['beta']
            self.forward_caches['z_tilde'] = z_tilde
            a = self.g(z_tilde)
        else:
            z += self.params['b']
            a = self.g(z)

        # dropout
        if self.dropout_keep_prob < 1.:
            a = inverted_dropout(a, self.dropout_keep_prob)

        self.forward_caches['a_pre'] = a_pre
        self.forward_caches['z'] = z
        self.forward_caches['a'] = a

        return a

    def backward(self, da: Optional[ndarray], y: ndarray) -> ndarray:
        batch_size = y.shape[SAMPLE_AXIS]
        if self.is_output_layer:
            dz = (self.forward_caches['a'] - y) / batch_size
            db = np.sum(dz, axis=SAMPLE_AXIS, keepdims=True)
            self.backward_caches['b'] = db
        elif self.neural_network.batch_normalization:
            z = self.forward_caches['z']
            mu = self.forward_caches['mu']
            sigma_square = self.forward_caches['sigma_square']
            z_white = self.forward_caches['z_white']
            z_tilde = self.forward_caches['z_tilde']

            # TODO batch normalization
            dz_tilde = da * self.g_prime(z_tilde)
            dgamma = np.sum(z_white.T @ dz_tilde, axis=SAMPLE_AXIS, keepdims=True)
            self.backward_caches['gamma'] = dgamma
            dbeta = np.sum(dz_tilde, axis=SAMPLE_AXIS, keepdims=True)
            self.backward_caches['beta'] = dbeta
            dz_white = dz_tilde @ self.params['gamma'].T
            ddsigma_square = -0.5 * (z - mu) * ((sigma_square + EPSILON)**(-1.5))
            ddmu = -1 / np.sqrt(sigma_square + EPSILON) - 2 / batch_size * np.sum(z - mu, axis=SAMPLE_AXIS, keepdims=True) * ddsigma_square
            ddz = 1 / np.sqrt(sigma_square + EPSILON) + ddsigma_square * (2 / batch_size) * (z - mu) + ddmu / batch_size
            dz = dz_white * ddz
            # dsigma_square = dz_white.T @ (z - mu) * -0.5 * ((sigma_square + EPSILON)**(-1.5))
            # dmu = -np.sum(dz_white, axis=SAMPLE_AXIS, keepdims=True) / np.sqrt(sigma_square + EPSILON) + -2 / batch_size * dsigma_square * np.sum(z - mu, axis=SAMPLE_AXIS, keepdims=True)
            # dz = dz_white / np.sqrt(sigma_square + EPSILON) + dsigma_square * (2 / batch_size) * (z - mu) + dmu / batch_size
        else:
            dz = da * self.g_prime(self.forward_caches['z'])
            db = np.sum(dz, axis=SAMPLE_AXIS, keepdims=True)
            self.backward_caches['b'] = db

        dw = self.forward_caches['a_pre'].T @ dz
        self.backward_caches['w'] = dw

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

        if self.neural_network.batch_normalization and not self.is_output_layer:
            self.params['gamma'] = np.ones(
                shape=(1, self.output_dim)
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
