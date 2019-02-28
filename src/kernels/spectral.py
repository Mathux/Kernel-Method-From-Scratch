from itertools import product

import numpy as np

from src.kernels.kernel import StringKernel


class SpectralKernel(StringKernel):
    def __init__(self, dataset=None, k=6, verbose=True):
        super(SpectralKernel, self).__init__(
            dataset=dataset, name="spectral", verbose=verbose)
        self.k = k
        self.mers = [(''.join(c)) for c in product('ACGT', repeat=self.k)]

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers))
        for j, mer in enumerate(self.mers):
            phi[j] += 1 * (mer in x)
        return phi


if __name__ == "__main__":
    from src.tools.dataloader import SeqData
    data = SeqData()
    kernel = SpectralKernel(data.train, k=3, verbose=True)
    K = kernel.K
