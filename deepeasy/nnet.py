"""Numpy 实现神经网络。

create:   2019-02-18
modified: 2019-02-26
"""

import sys
import time

import numpy as np

from .activations import get_activation_func
from .optimizations import get_gd
from .costs import get_cost_func
from .plot import plot_nn_history
from .types import *
from .env import *


class NeuralNetwork:
    """标准神经网络实现。

    神经网络结构：
    nn_architecture: List[Dict] = [
        {   # 第一层
            ‘input_dim’: int,  # 该层每个神经元被连接数
            'output_dim': int, # 该层神经元数
            'activation': str, # 该层激活函数，可选
            'dropout_keep_prob': float  # 该层 dropout 神经元保留概率，可选
        },
        {   # 第二层
            ...
        }
        ...
    ]
    """

    def __init__(self,
                 nn_architecture: List[Dict],
                 seed: Optional[int] = None) -> None:
        """
        :param nn_architecture: 要初始化的神经网络结构。
        :param seed: 初始化权重矩阵的随机数种子。
        """

        self.nn_architecture: List[Dict] = nn_architecture
        self.layers_count: int = len(nn_architecture)
        self.seed: Optional[int] = seed

        # 当训练时指定了归一化输入，此变量会被设为 True
        # 用于测试模型和预测时
        self.batch_normalization: bool = True

        # dropout 每层神经元保留概率
        self.dropout_keep_probs: Dict[str, float] = {}

        self.train_count: int = 0  # 记录训练次数
        self.history: Dict[str, List[float]] = {}  # cost 记录等

        self.params: Dict[str, ndarray] = {}  # 存放参数
        self.activation_funcs: Dict[str, Callable] = {}  # 存放激活函数，和其求导

        self.init_layers(seed)

    def train(self,
              x_train: ndarray,
              y_train: ndarray,
              epochs: int = 10000,
              *,
              new_train: bool = False,
              batch_size: int = 0,
              lr: float = 0.01,
              batch_normalization: bool = True,
              gd_name: str = 'sgd',
              momentum_beta: float = 0.9,
              rmsprop_beta: float = 0.999,
              l2_lambda: Optional[float] = None,
              dropout: bool = False,
              cost_func_name: str = 'cross_entropy') -> None:
        """
        :param x_train:
        :param y_train:
        :param epochs:
        :param new_train:
        :param batch_size:
        :param lr: 学习率
        :param batch_normalization:
        :param gd_name: 梯度下降的算法
        :param momentum_beta:
        :param rmsprop_beta:
        :param l2_lambda:
        :param dropout:
        :param cost_func_name:

        维度：

        w.shape = (每个神经元被连接数, 该层神经元数)
        b.shape = (1, 该层神经元数)
        x.shape = (样本数, 每个样本 feature 数)
        y.shape = (样本数, 每个样本 feature 数)
        """

        gd = get_gd(gd_name, lr, momentum_beta, rmsprop_beta)
        cost_func = get_cost_func(cost_func_name)

        if new_train:
            self.reset_params(keep_history=True)

        # 正则化输入
        self.batch_normalization = batch_normalization

        print(f'开始训练，迭代次数：{epochs}')

        start = time.time()

        for i in range(epochs):

            t: int = 0
            cost: float = 0.
            accuracy: float = 0.
            gd.reset()

            for x_batch, y_batch in self.mini_batch(x_train, y_train, batch_size):

                t += 1

                if batch_normalization:
                    x_batch = self.normalize_batch(x_batch)

                # 正向传播
                a, forward_caches = self.forward_propration(x_batch, dropout)

                # 计算误差，精度
                cost += cost_func(a, y_batch)
                accuracy += self.get_accuracy(a, y_batch)

                # 反向传播
                backward_caches = self.backward_propration(
                    a, y_batch, forward_caches
                )

                # 更新 w，b
                gd.update(self.params, backward_caches)

            self._add_to_history(cost=cost/t, accuracy=accuracy/t)

            sys.stdout.write(f'\r{i + 1}')
            sys.stdout.flush()

        end = time.time()
        print(f'\n完成！用时：{end - start}s')

    def test_model(self,
                   x: ndarray,
                   y: ndarray) -> float:

        if self.batch_normalization:
            x = self.normalize_batch(x)
        a = self.predict(x)
        return self.get_accuracy(a, y)

    def predict(self, x: ndarray) -> ndarray:

        x = np.atleast_2d(x)
        return self.forward_propration(x)[0]

    def reset_params(self,
                     keep_history: bool = False,
                     seed: Optional[int] = None) -> None:
        """重置神经网络的参数。用于验证另一组超参。

        :param keep_history: 是否保存从前历史。
        :param seed: 随机数种子。
        """

        if seed:
            np.random.seed(seed)
        else:
            np.random.seed(self.seed)  # 使用和前一次一样的随机数种子

        if not keep_history:
            self.history.clear()
            # 重置训练计数
            self.train_count = 0

        # 代表新一轮训练
        if self.history:  # 判断是否已有训练数据，否则不能加 1
            self.train_count += 1

        # 清空原来的参数
        self.params.clear()
        self.batch_normalization = True

        for layer_idx, layer in enumerate(self.nn_architecture, 1):
            self._init_per_layer_params(layer_idx, layer)

    def init_layers(self, seed: Optional[int] = None) -> None:

        # 随机数种子
        np.random.seed(seed)

        for layer_idx, layer in enumerate(self.nn_architecture, 1):
            self._init_per_layer_params(layer_idx, layer)

            g, g_backward = get_activation_func(layer.get('activation'))
            self.activation_funcs[f'g{layer_idx}'] = g
            self.activation_funcs[f'g_prime{layer_idx}'] = g_backward

            # 初始化每层的 dropout 的 keep prob
            kp = layer.get('dropout_keep_prob')
            self.dropout_keep_probs[f'kp{layer_idx}'] = kp if kp else 1.

    def normalize_batch(self, x: ndarray) -> ndarray:
        """归一化。"""

        # samples_num = x.shape[SMAPLES_NUM_AXIS]
        # mu = np.sum(x, SMAPLES_NUM_AXIS, keepdims=True) / samples_num
        # self.normalize_params['mu'] = mu
        # x = x - mu
        # sigma_square = np.sum(x**2, SMAPLES_NUM_AXIS, keepdims=True) / samples_num
        # self.normalize_params['sigma_square'] = sigma_square
        # return x / np.sqrt(sigma_square+DELTA)
        return x / 255.

    @staticmethod
    def mini_batch(x_train: ndarray,
                   y_train: ndarray,
                   batch_size: int) -> Iterator[Tuple[ndarray, ndarray]]:
        """每次迭代随机抽 batch_size 的样本出来。"""

        samples_num = x_train.shape[LABELS_NUM_AXIS]
        if batch_size <= 0:
            batch_size = samples_num

        permutation = np.random.permutation(samples_num)
        x_shuffle = x_train[permutation]
        y_shuffle = y_train[permutation]
        end_flag: bool = False
        for j in range(0, samples_num, batch_size):
            end = j + batch_size
            remain = samples_num - end
            if 0 < remain < batch_size:
                end += remain
                end_flag = True

            x_batch = x_shuffle[j:end]
            y_batch = y_shuffle[j:end]

            yield x_batch, y_batch

            if end_flag:
                return

    def forward_propration(self,
                           x: ndarray,
                           dropout: bool = False) -> Tuple[ndarray, Dict]:

        a = x
        forward_caches = {'a0': a}

        for layer_idx in range(1, self.layers_count + 1):
            w = self.params[f'w{layer_idx}']
            b = self.params[f'b{layer_idx}']
            activation_func = self.activation_funcs[f'g{layer_idx}']

            z = a @ w + b
            a = activation_func(z)  # shape = (该层神经元数, 样本数)

            # dropout
            if dropout:
                kp = self.dropout_keep_probs[f'kp{layer_idx}']
                if kp < 1.:
                    a = self.inverted_dropout(a, kp)

            forward_caches[f'z{layer_idx}'] = z
            forward_caches[f'a{layer_idx}'] = a

        return a, forward_caches

    def backward_propration(self,
                            a: ndarray,
                            y: ndarray,
                            forward_caches: Dict) -> Dict:

        backward_caches = {}

        batch_size = a.shape[LABELS_NUM_AXIS]

        # da = -(y / a) + (1 - y) / (1 - a)
        da: Optional[ndarray] = None

        for layer_idx in range(self.layers_count, 0, -1):
            activation_func_prime = self.activation_funcs[f'g_prime{layer_idx}']
            z = forward_caches[f'z{layer_idx}']
            a_pre = forward_caches[f'a{layer_idx - 1}']
            w = self.params[f'w{layer_idx}']
            if layer_idx == self.layers_count:
                dz = (a - y) / batch_size
            else:
                dz = da * activation_func_prime(z)

            dw = (a_pre.T @ dz)

            db = np.sum(dz, axis=LABELS_NUM_AXIS, keepdims=True)

            backward_caches[f'w{layer_idx}'] = dw
            backward_caches[f'b{layer_idx}'] = db

            da = dz @ w.T

        return backward_caches

    # def l2_regularization(self, l2_lambda: float) -> float:
    #     all_w_sum = 0
    #     for layer_idx in range(1, self.layers_count + 1):
    #         all_w_sum += np.sum(self.params_values[f'w{layer_idx}'] ** 2)
    #     return l2_lambda / 2 * all_w_sum

    @staticmethod
    def inverted_dropout(a: ndarray,
                         keep_prob: float,
                         seed: Optional[int] = None) -> ndarray:

        np.random.seed(seed)
        d = np.random.rand(*a.shape) < keep_prob
        return a * d / keep_prob

    @staticmethod
    def get_accuracy(a: ndarray, y: ndarray) -> float:
        feature_num = a.shape[LABEL_FEATURES_NUM_AXIS]
        batch_size = a.shape[LABELS_NUM_AXIS]

        if feature_num == 1:
            a = np.where(a > 0.5, 1, 0)
            return np.sum(a == y) / batch_size
        else:
            return np.sum(
                a.argmax(axis=LABEL_FEATURES_NUM_AXIS)
                == y.argmax(axis=LABEL_FEATURES_NUM_AXIS)
            ) / batch_size

    def plot_history(self):
        plot_nn_history(self)

    def _init_per_layer_params(self, layer_idx: int, layer: Dict) -> None:
        """初始化权重和偏差。"""

        layer_input_size = layer['input_dim']
        layer_output_size = layer['output_dim']
        act = layer.get('activation')  # 可能为 None

        # Xavier Initializer
        # 除以每个神经元被连接数的平方根
        # 如果用 ReLU，除以每个神经元被连接数的平方根乘 2，会更好
        n = layer_input_size
        if act and act.lower() == 'relu':
            n /= 2
        self.params[f'w{layer_idx}'] = np.random.randn(
            layer_input_size, layer_output_size
        ) / np.sqrt(n)

        # b 全初始化为 0
        self.params[f'b{layer_idx}'] = np.zeros(
            shape=(1, layer_output_size)
        )

    def _add_to_history(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            self.history.setdefault(f'{k}{self.train_count}', []).append(v)
