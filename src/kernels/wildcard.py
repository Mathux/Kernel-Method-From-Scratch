#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:04:38 2019

@author: evrardgarcelon, mathispetrovich
"""

import numpy as np
from src.kernels.kernel import StringKernel, KernelCreate


def wildcard_match(x, y):
    result = True
    for i in range(len(x)):
        if x[i] != y[i] and y[i] != '*':
            result = False
            break
    return result


class WildcardKernel(StringKernel, metaclass=KernelCreate):
    name = "wildcard"
    defaultParameters = {"k": 2, "m": 1, 'la': 1}

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers_wildcard))
        for i in range(len(x) - self.param.k + 1):
            x_kmer = x[i:i + self.param.k]
            for j, b in enumerate(self.mers_wildcard):
                phi[j] += self.param.la**(b.count('*')) * wildcard_match(
                    x_kmer, b)
        return phi


if __name__ == "__main__":
    dparams = {"small": True, "nsmall": 100}
    kparams = {'k': 5, 'm': 1, 'la': 1}

    from src.tools.test import EasyTest
    EasyTest(kernel="wildcard", data="seq", dparams=dparams, kparams=kparams)


    """
    from src.data.seq import SeqData
    data = SeqData(small=False)
    data.data = np.array([data.data[0]])
    import time
    debut = time.perf_counter()
    kernel = WildcardKernel(data, parameters=
    fin = time.perf_counter()
    print(fin - debut)
    from src.methods.kpca import KPCA
    kpca = KPCA(kernel)
    proj = kpca.project()

    data.show_pca(proj)
"""
