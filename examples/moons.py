from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

import deepeasy.nnet as nnet
from deepeasy.log import logger

N_SAMPLES = 1000
NOISE = 0.25
SEED = 100

# ratio between training and test sets
TEST_SIZE = 0.1


def main() -> None:
    x, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TEST_SIZE, random_state=42)
    y_train= y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    logger.info(f'x_train.shape={x_train.shape}, y_train.shape={y_train.shape}')
    logger.info(f'x_test.shape={x_test.shape}, y_test.shape={y_test.shape}')

    nn_architecture = [
        {'input_dim': 2, 'output_dim': 16, 'activation': 'relu'},
        {'input_dim': 16, 'output_dim': 16, 'activation': 'relu'},
        {'input_dim': 16, 'output_dim': 1, 'activation': 'sigmoid'},
    ]
    nn = nnet.NeuralNetwork(nn_architecture, seed=100)
    nn.train(
        x_train, y_train, 1000,
        new_train=True,
        batch_size=100,
        lr=0.001,
        batch_normalization=False,
        optimizer_name='adam'
    )
    logger.info(nn.test_model(x_test, y_test))
    nn.plot_history()


if __name__ == '__main__':
    main()
