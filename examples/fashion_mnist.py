"""fashion-mnist 识别。"""

import deepeasy.nnet as nnet
from deepeasy.log import logger
from deepeasy.datasets.mnist import load_mnist, show_mnist


def main() -> None:
    # 需要提前下好，放入同一个文件夹
    # 下载地址：https://github.com/zalandoresearch/fashion-mnist
    # 一共 4 个 *.gz 文件
    # 分别代表训练数据、训练数据标签、测试数据、测试数据标签
    file_path = '/home/zzzzer/Documents/data/数据集/fashion_mnist/'
    x_train, y_train, x_test, y_test = load_mnist(file_path)
    logger.info(f'x_train.shape={x_train.shape}, y_train.shape={y_train.shape}')
    logger.info(f'x_test.shape={x_test.shape}, y_test.shape={y_test.shape}')

    # 大概显示图片，需安装 PIL
    # show_mnist(x_train)

    # 神经网络结构
    nn_architecture = [
        {'input_dim': 28 * 28, 'output_dim': 64, 'activation': 'relu'},
        {'input_dim': 64, 'output_dim': 64, 'activation': 'relu'},
        {'input_dim': 64, 'output_dim': 10, 'activation': 'softmax'},
    ]

    nn = nnet.NeuralNetwork(nn_architecture, batch_normalization=False, seed=100)

    nn.train(
        x_train, y_train, 10,
        new_train=True,
        batch_size=600,
        lr=0.001,
        optimizer_name='adam'
    )

    logger.info(nn.test_model(x_test, y_test))
    nn.plot_history()


if __name__ == '__main__':
    main()
