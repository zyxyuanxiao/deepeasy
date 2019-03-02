import deepeasy.nnet as nnet
from deepeasy.datasets import load_mnist


def main() -> None:
    file_path = '/home/zzzzer/Documents/data/数据集/数字手写体/mnist/'
    x_train, y_train, x_test, y_test = load_mnist(file_path)

    # 神经网络结构
    nn_architecture = [
        {"input_dim": 28 * 28, "output_dim": 50, "activation": "relu"},
        {"input_dim": 50, "output_dim": 10, "activation": "softmax"},
    ]

    nn = nnet.NeuralNetwork(nn_architecture, 55)

    nn.train(
        x_train, y_train, 100,
        new_train=True,
        batch_size=600,
        lr=0.01,
        gd_name='sgd'
    )

    print(nn.test_model(x_test, y_test))


if __name__ == '__main__':
    main()
