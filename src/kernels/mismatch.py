import numpy as np

from src.kernels.kernel import StringKernel, KernelCreate
from src.tools.utils import nb_diff


class MismatchKernel(StringKernel, metaclass=KernelCreate):
    name = "mismatch"
    defaultParameters = {"k": 3, "m": 1}

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers))
        for i in range(len(x) - self.param.k + 1):
            x_kmer = x[i:i + self.param.k]
            for j, b in enumerate(self.mers):
                phi[j] += 1 * (nb_diff(x_kmer, b) <= self.param.m)
        return phi


if __name__ == "__main__":
    dparams = {"small": True, "nsmall": 100}
    kparams = {"k": 5, "m": 2}

    from src.tools.test import EasyTest
    EasyTest(kernel="mismatch", data="seq", dparams=dparams, kparams=kparams)
