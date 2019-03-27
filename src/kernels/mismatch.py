import numpy as np
from src.kernels.kernel import StringKernel, TrieKernel, SparseKernel
from src.kernels.kernel import KernelCreate
from src.tools.utils import nb_diff
from src.tools.utils import Parameters
from src.data.trie_dna import MismatchTrie


class MismatchStringKernel(StringKernel, metaclass=KernelCreate):
    name = "mismatch"
    defaultParameters = {"k": 3, "m": 1, "trie": False, "sparse": False}

    def _compute_phi(self, x):
        phi = np.zeros(len(self.mers))
        for i in range(len(x) - self.param.k + 1):
            x_kmer = x[i:i + self.param.k]
            for j, b in enumerate(self.mers):
                phi[j] += 1 * (nb_diff(x_kmer, b) <= self.param.m)
        return phi


class MismatchTrieKernel(TrieKernel, metaclass=KernelCreate):
    name = "mismatch"
    defaultParameters = {"k": 3, "m": 1, "trie": True, "sparse": False}
    Trie = MismatchTrie


class MismatchSparseKernel(SparseKernel, metaclass=KernelCreate):
    name = "mismatch"
    defaultParameters = {"k": 3, "m": 1, "trie": False, "sparse": True}

    def _compute_phi(self, x):
        phi = {}
        for _, b in enumerate(self.mers):
            for i in range(len(x) - self.param.k + 1):
                xkmer = x[i:i + self.param.k]
                phi[b] = phi.get(b,
                                 0) + 1 * (nb_diff(xkmer, b) <= self.param.m)
        return phi


class __MismatchKernel:
    def __init__(self):
        self.defaultParameters = {
            "k": 5,
            'm': 1,
            "trie": False,
            "sparse": True
        }
        self.name = "mismatch"

    def __call__(self, dataset=None, parameters=None, verbose=True):
        param = Parameters(parameters, self.defaultParameters)
        if param.sparse:
            return MismatchSparseKernel(dataset, parameters, verbose)
        else:
            if param.trie:
                return MismatchTrieKernel(dataset, parameters, verbose)
            else:
                return MismatchStringKernel(dataset, param, verbose)


MismatchKernel = __MismatchKernel()

if __name__ == "__main__":
    dparams = {"small": True, "nsmall": 100}
    kparams = {"k": 5, "m": 2}

    # from src.tools.test import EasyTest
    # EasyTest(kernels="mismatch", data="seq", dparams=dparams, kparams=kparams)

    from src.tools.test import KernelTest

    parameters = []
    K = 6
    # parameters.append({"k": K, "sparse": True, "trie": False})
    parameters.append({"k": K, "sparse": False, "trie": False})
    # parameters.append({"k": K, "sparse": False, "trie": True})
    KernelTest("mismatch", parameters)
