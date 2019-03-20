import numpy as np
from src.kernels.kernel import StringKernel, KernelCreate
from src.kernels.wildcard import WildcardTrieKernel
from src.tools.utils import Parameters


class SpectralStringKernel(StringKernel, metaclass=KernelCreate):
    name = "spectral"
    defaultParameters = {"k": 6, "trie": False}

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers))
        for offset in range(len(x)) :
            xkmer = x[offset : offset + self.param.k]
            for j, mer in enumerate(self.mers):
                phi[j] += 1 * (xkmer == mer)
        return phi


class __SpectralKernel:
    def __init__(self):
        self.defaultParameters = {"k": 6, "trie": True}
        self.__trieParameters = {"k": 6, 'm': 0, 'la': 1, "trie": True}
        self.name = "spectral"

    def __call__(self, dataset=None, parameters=None, verbose=True):
        param = Parameters(parameters, self.defaultParameters)
        if param.trie:
            parameters = Parameters(parameters, self.__trieParameters)
            return WildcardTrieKernel(dataset, parameters, verbose)
        else:
            return SpectralStringKernel(dataset, param, verbose)


SpectralKernel = __SpectralKernel()


if __name__ == "__main__":
    dparams = {"small": False, "nsmall": 100}
    kparams = {"k": 6, "m": 0, "trie": True}
    from src.tools.test import EasyTest
    EasyTest(kernels="wildcard", data="seq", dparams=dparams, kparams=kparams)
