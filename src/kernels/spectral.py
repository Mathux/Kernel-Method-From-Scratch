import numpy as np
from src.kernels.kernel import StringKernel, SparseKernel, KernelCreate
from src.kernels.mismatch import MismatchTrieKernel
from src.tools.utils import Parameters


class SpectralStringKernel(StringKernel, metaclass=KernelCreate):
    name = "spectral"
    defaultParameters = {"k": 6, "trie": False, "sparse": False}

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers))
        for offset in range(len(x)):
            xkmer = x[offset:offset + self.param.k]
            for j, mer in enumerate(self.mers):
                phi[j] += 1 * (xkmer == mer)
        return phi


class SpectralSparseKernel(SparseKernel, metaclass=KernelCreate):
    name = "spectral"
    defaultParameters = {"k": 6, "trie": False, "sparse": True}

    def _compute_phi(self, x):
        phi = {}
        for offset in range(len(x) - self.param.k + 1):
            xkmer = x[offset:offset + self.param.k]
            phi[xkmer] = phi.get(xkmer, 0) + 1
        return phi


class __SpectralKernel:
    def __init__(self):
        self.defaultParameters = {"k": 6, "trie": False, "sparse": True}
        self.__trieParameters = {
            "k": 6,
            'm': 0,
            "trie": True,
            "sparse": False
        }
        self.name = "spectral"

    def __call__(self, dataset=None, parameters=None, verbose=True):
        param = Parameters(parameters, self.defaultParameters)
        if param.sparse:
            return SpectralSparseKernel(dataset, param, verbose)
        else:
            if param.trie:
                parameters = Parameters(parameters, self.__trieParameters)
                return MismatchTrieKernel(dataset, parameters, verbose)
            else:
                return SpectralStringKernel(dataset, param, verbose)


SpectralKernel = __SpectralKernel()


class SpectralConcatKernel(SparseKernel, metaclass=KernelCreate):
    name = "spectralconcat"
    defaultParameters = {"kmin": 6, "kmax": 11, "lam": 1.0}

    def _compute_phi(self, x):
        phi = {}
        for k in range(self.param.kmin, self.param.kmax + 1):
            count = self.param.lam**(self.param.kmax - k)
            for offset in range(len(x) - k + 1):
                xkmer = x[offset:offset + k]
                phi[xkmer] = phi.get(xkmer, 0) + count
        return phi


if __name__ == "__main__":
    dparams = {"small": False, "nsmall": 100}
    kparams = {"k": 12, "sparse": True}
    # from src.tools.test import EasyTest
    # EasyTest(kernels="spectral", data="seq", dparams=dparams, kparams=kparams)
    from src.tools.test import KernelTest
    parameters = []

    kmin = 5
    kmax = 6
    parameters.append({"kmin": kmin, "kmax": kmax})
    KernelTest("spectralconcat", parameters)
    """K = 6
    parameters.append({"k": K})
    KernelTest("spectral", parameters)"""
    
"""K = 2
    parameters.append({"k": K, "sparse": False, "trie": False})
    parameters.append({"k": K, "sparse": False, "trie": True})
    parameters.append({"k": K, "sparse": True, "trie": False})
    KernelTest("spectral", parameters)
"""
