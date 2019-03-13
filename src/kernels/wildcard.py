#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:04:38 2019

@author: evrardgarcelon
"""

import numpy as np

from src.kernels.kernel import StringKernel, KernelCreate


class WildcardKernel(StringKernel, metaclass=KernelCreate):
    defaultParameters = {"k": 5, "m": 1, 'la': 1}

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers_wildcard))
        for i in range(len(x) - self.param.k + 1):
            x_kmer = x[i:i + self.param.k]
            for j, b in enumerate(self.mers_wildcard):

                phi[j] += self.param.la**(b.count('*')) * (x_kmer == b.replace(
                    '*', ''))
        return phi


if __name__ == "__main__":
    from src.data.seq import SeqData
    data = SeqData(small=False)

    kernel = WildcardKernel(data)

    from src.methods.kpca import KPCA
    kpca = KPCA(kernel)
    proj = kpca.project()

    data.show_pca(proj)
