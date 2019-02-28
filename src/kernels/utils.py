import numpy as np


# Normalise the kernel (divide by the variance)
def normalize_kernel(K):
    nkernel = np.copy(K)

    assert nkernel.ndim == 2
    assert nkernel.shape[0] == nkernel.shape[1]

    for i in range(nkernel.shape[0]):
        for j in range(i + 1, nkernel.shape[0]):
            q = np.sqrt(nkernel[i, i] * nkernel[j, j])
            if q > 0:
                nkernel[i, j] /= q
                nkernel[j, i] = nkernel[i, j]
    np.fill_diagonal(nkernel, 1.)

    return nkernel


def nb_diff(x, y):
    nb_diff = 0
    for char1, char2 in zip(x, y):
        if char1 != char2:
            nb_diff += 1
    return nb_diff


def get_kernel_parameters(kernel):
    if kernel == 'linear':
        return ['offset']
    elif kernel == 'gaussian':
        return ['gamma']
    elif kernel == 'sigmoid':
        return ['gamma', 'offset']
    elif kernel == 'polynomial':
        return ['offset', 'dim']
    elif kernel == 'laplace':
        return ['gamma']
    elif kernel == 'spectral':
        return ['k']
    elif kernel == 'mismatch':
        return ['k', 'm']
    elif kernel == 'WD':
        return ['d']
    elif kernel == 'laplace':
        return ['gamma']
    else:
        raise Exception('Invalid Kernel')

