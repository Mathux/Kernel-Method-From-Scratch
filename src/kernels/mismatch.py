import numpy as np
from src.kernels.kernel import StringKernel, TrieKernel, KernelCreate
from src.tools.utils import nb_diff
from src.tools.utils import Parameters
from src.data.trie_dna import MismatchTrie


class MismatchStringKernel(StringKernel, metaclass=KernelCreate):
    name = "mismatch"
    defaultParameters = {"k": 3, "m": 1, "trie": False}

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers))
        for i in range(len(x) - self.param.k + 1):
            x_kmer = x[i:i + self.param.k]
            for j, b in enumerate(self.mers):
                phi[j] += 1 * (nb_diff(x_kmer, b) <= self.param.m)
        return phi


class MismatchTrieKernel(TrieKernel, metaclass=KernelCreate):
    name = "mismatch"
    defaultParameters = {"k": 3, "m": 1, "trie": True}
    Trie = MismatchTrie


class __MismatchKernel:
    def __init__(self):
        self.defaultParameters = {"k": 5, 'm': 1, "trie": True}
        self.name = "mismatch"

    def __call__(self, dataset=None, parameters=None, verbose=True):
        param = Parameters(parameters, self.defaultParameters)
        if param.trie:
            return MismatchTrieKernel(dataset, parameters, verbose)
        else:
            return MismatchStringKernel(dataset, param, verbose)
        

MismatchKernel = __MismatchKernel()

if __name__ == "__main__":
    dparams = {"small": True, "nsmall": 100}
    kparams = {"k": 5, "m": 2}

    from src.tools.test import EasyTest
    EasyTest(kernel="mismatch", data="seq", dparams=dparams, kparams=kparams)
