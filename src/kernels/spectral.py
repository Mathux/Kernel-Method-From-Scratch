import numpy as np
from src.kernels.kernel import StringKernel, KernelCreate


class SpectralKernel(StringKernel, metaclass=KernelCreate):
    name = "spectral"
    defaultParameters = {"k": 6}

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers))
        for offset in range(len(x)) :
            xkmer = x[offset : offset + self.param.k]
            for j, mer in enumerate(self.mers):
                phi[j] += 1 * (xkmer == mer)
        return phi


if __name__ == "__main__":
    dparams = {"small": True, "nsmall": 300}
    kparams = {"k": 6}

    from src.tools.test import EasyTest
    EasyTest(kernel="spectral", data="seq", dparams=dparams, kparams=kparams)
