import numpy as np
from ..types import *


def load_mnist(path: str) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    :param path: mnist 数据所在文件夹路径
    """

    x_train = load_mnist_images(path+'train-images-idx3-ubyte')
    y_train = load_mnist_label(path+'train-labels-idx1-ubyte')
    x_test = load_mnist_images(path+'t10k-images-idx3-ubyte')
    y_test = load_mnist_label(path+'t10k-labels-idx1-ubyte')
    x_train = x_train.reshape(-1, 28*28)
    y_train = change_one_hot_label(y_train)
    x_test = x_test.reshape(-1, 28 * 28)
    y_test = change_one_hot_label(y_test)
    return x_train, y_train, x_test, y_test


def load_mnist_label(path: str) -> ndarray:
    with open(path, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.int8, offset=8)


def load_mnist_images(path: str) -> ndarray:
    with open(path, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)


def change_one_hot_label(y: ndarray) -> ndarray:
    y_temp = np.zeros((y.shape[0], 10), dtype=np.int8)
    for i, v in enumerate(y):
        y_temp[i, v] = 1
    return y_temp
