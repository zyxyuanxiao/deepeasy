import numpy as np

from deepeasy.normalization import whitening_train
from deepeasy.env import EPSILON


def test_whitening():
    a = np.array([[1, 2, 3], [7, 6, 5]])
    a_white, mu, sigma_square = whitening_train(a)
    assert np.equal(mu, a.mean(axis=0, keepdims=True)).all()
    assert np.equal(sigma_square, a.var(axis=0, keepdims=True)).all()
    assert np.equal(a_white, (a - mu)/np.sqrt(sigma_square+EPSILON)).all()
