import numpy as np


def quad(K, alpha):
    return np.dot(alpha, np.dot(K, alpha))
