import numpy as np

    
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

