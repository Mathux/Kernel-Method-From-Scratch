from itertools import product

import numpy as np

from src.kernels.kernel import StringKernel
from src.kernels.utils import nb_diff


class MismatchKernel(StringKernel):
    def __init__(self, dataset=None, k=3, m=1, verbose=True):
        super(MismatchKernel, self).__init__(
            dataset=dataset, name="mismatch", verbose=verbose)
        self.k = k
        self.m = m
        self.mers = [(''.join(c)) for c in product('ACGT', repeat=self.k)]

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers))
        for i in range(len(x) - self.k + 1):
            x_kmer = x[i:i + self.k]
            for j, b in enumerate(self.mers):
                phi[j] += 1 * (nb_diff(x_kmer, b) <= self.m)

        return phi


if __name__ == "__main__":
    from src.tools.dataloader import SeqData
    data = SeqData()
    kernel = MismatchKernel(data.train, k=3, m=1, verbose=True)
    K = kernel.K
