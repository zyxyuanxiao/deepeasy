"""Numpy 实现神经网络。

create:   2019-02-18
modified: 2019-02-26
"""

import sys
import time

import numpy as np

from .layers import Layer
from .optimizers import get_gd
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

        self.layers_count: int = len(nn_architecture)
        self.layers: List[Layer] = []

        self.seed: Optional[int] = seed

        self.batch_normalization: bool = True

        self.train_count: int = 0  # 记录训练次数
        self.history: Dict[str, List[float]] = {}  # cost 记录等

        self.init_layers(nn_architecture, seed)

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
              beta1: float = 0.9,
              beta2: float = 0.999,
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
        :param beta1:
        :param beta2:
        :param l2_lambda:
        :param dropout:
        :param cost_func_name:

        维度：

        w.shape = (每个神经元被连接数, 该层神经元数)
        b.shape = (1, 该层神经元数)
        x.shape = (样本数, 每个样本 feature 数)
        y.shape = (样本数, 每个样本 feature 数)
        """

        gd = get_gd(gd_name, lr, beta1, beta2)
        cost_func = get_cost_func(cost_func_name)

        if new_train:
            self.reset_params(keep_history=True)

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

                # 正向传播
                a = self.forward_propration(x_batch)

                # 计算误差，精度
                cost += cost_func(a, y_batch)
                accuracy += self.get_accuracy(a, y_batch)

                # 反向传播
                self.backward_propration(y_batch)

                # 更新 w，b
                gd.update(self.layers)

            self._add_to_history(cost=cost/t, accuracy=accuracy/t)

            sys.stdout.write(f'\r{i + 1}')
            sys.stdout.flush()

        end = time.time()
        print(f'\n完成！用时：{end - start}s')

    def test_model(self, x: ndarray, y: ndarray) -> float:

        if self.batch_normalization:
            x = self.normalize(x)
        a = self.predict(x)
        return self.get_accuracy(a, y)

    def predict(self, x: ndarray) -> ndarray:

        x = np.atleast_2d(x)
        return self.forward_propration(x)

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

        self.batch_normalization = True

        for layer in self.layers:
            layer.init_params()

    def init_layers(self,
                    nn_architecture: List[Dict],
                    seed: Optional[int] = None) -> None:

        # 随机数种子
        np.random.seed(seed)

        for layer_arch in nn_architecture:
            input_dim = layer_arch['input_dim']
            output_dim = layer_arch['output_dim']
            activation = layer_arch.get('activation')  # 可能为 None
            dropout_keep_prob = layer_arch.get('dropout_keep_prob', 1.)

            layer = Layer(
                input_dim,
                output_dim,
                activation,
                dropout_keep_prob
            )

            self.layers.append(layer)

    def normalize(self, x: ndarray) -> ndarray:
        """归一化。"""

        # TODO
        pass

    @staticmethod
    def whitening(z: ndarray) -> Tuple[ndarray, float, float]:
        samples_num = z.shape[LABELS_NUM_AXIS]
        mu = np.sum(z, LABELS_NUM_AXIS, keepdims=True) / samples_num
        x = z - mu
        sigma_square = np.sum(x**2, LABELS_NUM_AXIS, keepdims=True) / samples_num
        return x / np.sqrt(sigma_square + EPSILON), mu, sigma_square

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

    def forward_propration(self, x: ndarray) -> ndarray:
        a = x
        for layer in self.layers:
            a = layer.forward(a)
        return a

    def backward_propration(self, y: ndarray) -> None:
        da: Optional[ndarray] = None
        for layer in reversed(self.layers):
            da = layer.backward(da, y)

    # def l2_regularization(self, l2_lambda: float) -> float:
    #     all_w_sum = 0
    #     for layer_idx in range(1, self.layers_count + 1):
    #         all_w_sum += np.sum(self.params_values[f'w{layer_idx}'] ** 2)
    #     return l2_lambda / 2 * all_w_sum

    @staticmethod
    def get_accuracy(a: ndarray, y: ndarray) -> float:
        feature_num = y.shape[LABEL_FEATURES_NUM_AXIS]
        batch_size = y.shape[LABELS_NUM_AXIS]

        if feature_num == 1:
            a = np.where(a > 0.5, 1, 0)
            return np.sum(a == y) / batch_size
        else:
            return np.sum(
                a.argmax(axis=LABEL_FEATURES_NUM_AXIS)
                == y.argmax(axis=LABEL_FEATURES_NUM_AXIS)
            ) / batch_size

    def plot_history(self) -> None:
        plot_nn_history(self)

    def _add_to_history(self, **kwargs: float) -> None:
        for k, v in kwargs.items():
            self.history.setdefault(f'{k}{self.train_count}', []).append(v)
