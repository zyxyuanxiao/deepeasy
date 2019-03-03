import deepeasy.nnet as nnet
from deepeasy.log import logger
from deepeasy.datasets import load_mnist


def main() -> None:
    file_path = '/home/zzzzer/Documents/data/数据集/数字手写体/mnist/'
    x_train, y_train, x_test, y_test = load_mnist(file_path)
    logger.info(f'x_train.shape={x_train.shape}, y_train.shape={y_train.shape}')
    logger.info(f'x_test.shape={x_test.shape}, y_test.shape={y_test.shape}')

    # 神经网络结构
    nn_architecture = [
        {"input_dim": 28 * 28, "output_dim": 64, "activation": "relu"},
        {"input_dim": 64, "output_dim": 10, "activation": "softmax"},
    ]

    nn = nnet.NeuralNetwork(nn_architecture, batch_normalization=False, seed=100)

    nn.train(
        x_train, y_train, 100,
        new_train=True,
        batch_size=600,
        lr=0.016,
        gd_name='sgd'
    )

    logger.info(nn.test_model(x_test, y_test))
    nn.plot_history()


if __name__ == '__main__':
    main()
