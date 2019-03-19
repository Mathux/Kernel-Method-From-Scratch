import numpy as np
from src.kernels.kernel import StringKernel, TrieKernel, KernelCreate
from src.tools.utils import nb_diff
from src.tools.utils import Parameters


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


raise NotImplementedError("k_value fonction?")
class MismatchTrieKernel(TrieKernel, metaclass=KernelCreate):
    name = "mismatch"
    defaultParameters = {"k": 3, "m": 1, "trie": True}

    def k_value(self, x, changev=False):
        leafs = self.get_leaf_nodes(self.trie)

        oldvalue = self.verbose
        if changev:
            self.verbose = False
            
        desc = "Leaf computation"
        self.leaf_kgrams_ = dict((leaf.full_label,
                                  dict((index, (len(kgs),
                                                leaf.full_label.count('*')))
                                       for index, kgs in leaf.kgrams.items()))
                                 for leaf in self.viterator(leafs, desc=desc))
        k_x = np.zeros(len(self.data))

        self.verbose = oldvalue
                
        for kmer, count1 in self.unique_kmers(x):
            if kmer in list(self.leaf_kgrams_.keys()):
                for j in range(len(self.data.data)):
                    if j in list(self.leaf_kgrams_[kmer].keys()):
                        kgrams, nb_wildcard = self.leaf_kgrams_[kmer][j]
                        k_x[j] += self.param.la**nb_wildcard * (
                            count1 * kgrams)

        return k_x


class __MismatchKernel:
    def __init__(self):
        self.defaultParameters = {"k": 5, 'm': 1, 'la': 1, "trie": True}
        self.name = "wildcard"

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
