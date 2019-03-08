"""
下载地址：
1、手写体数字：http://yann.lecun.com/exdb/mnist/
2、fashion-mnist：https://github.com/zalandoresearch/fashion-mnist
"""

import gzip

import numpy as np
from ..log import logger
from ..mytypes import *


def load_mnist(path: str) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    :param path: mnist 数据所在文件夹路径
    """

    x_train = load_mnist_images(path+'train-images-idx3-ubyte.gz')
    y_train = load_mnist_label(path+'train-labels-idx1-ubyte.gz')
    x_test = load_mnist_images(path+'t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_label(path+'t10k-labels-idx1-ubyte.gz')
    x_train = x_train.reshape(-1, 28 * 28)
    y_train = change_one_hot_label(y_train)
    x_test = x_test.reshape(-1, 28 * 28)
    y_test = change_one_hot_label(y_test)
    return x_train, y_train, x_test, y_test


def load_mnist_label(path: str) -> ndarray:
    with gzip.open(path, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.int8, offset=8)


def load_mnist_images(path: str) -> ndarray:
    with gzip.open(path, 'rb') as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(-1, 28, 28)


def change_one_hot_label(y: ndarray) -> ndarray:
    y_temp = np.zeros((y.shape[0], 10), dtype=np.int8)
    for i, v in enumerate(y):
        y_temp[i, v] = 1
    return y_temp


def show_mnist(data, x_size: int = 16, y_size: int = 9) -> None:
    """
    需先安装 PIL，并且 *nix 需安装 imagemagick，或者自己修改
    *lib/python3.7/site-packages/PIL/ImageShow.py* 文件，
    添加其他图片查看器。

    :param data:
    :param x_size:
    :param y_size:
    :return:
    """

    try:
        from PIL import Image
    except ImportError:
        logger.error('需先安装 PIL！')
        return

    imgs = Image.new('RGBA', (30 * x_size, 30 * y_size))  # 打底背景
    for i in range(x_size*y_size):
        img = Image.fromarray(data[i].reshape(28, 28))
        imgs.paste(img, (i % x_size * 30, i // x_size * 30))  # 在打底背景上铺设照片
    imgs.show()
