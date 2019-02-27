from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import deepeasy.nnet as nnet

N_SAMPLES = 1000
NOISE = 0.22
SEED = 100

# ratio between training and test sets
TEST_SIZE = 0.1


def main() -> None:
    x, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    y_train= y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    print(f'x_train.shape={x_train.shape}, y_train.shape={y_train.shape}')
    print(f'x_test.shape={x_test.shape}, y_test.shape={y_test.shape}')

    nn_architecture = [
        {"input_dim": 2, "output_dim": 50, "activation": "relu"},
        {"input_dim": 50, "output_dim": 1, "activation": "sigmoid"},
    ]
    nn = nnet.NeuralNetwork(nn_architecture, 55)
    nn.train(
        x_train, y_train, 1000,
        new_train=True,
        batch_size=100,
        lr=0.1,
        gd_name='sgd'
    )


if __name__ == '__main__':
    main()
