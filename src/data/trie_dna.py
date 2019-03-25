import numpy as np
from src.tools.utils import Logger
# import ipdb


class _Trie(Logger):
    def is_leaf(self):
        return len(self.children) == 0

    def is_empty(self):
        return len(self.kgrams) == 0

    def copy_kgrams(self):
        return {
            index: np.array(substring_pointers)
            for index, substring_pointers in self.kgrams.items()
        }

    def add_child(self, child):
        assert child.label not in self.children
        child.kgrams = self.copy_kgrams()
        child.level = self.level + 1
        self.children[child.label] = child
        child.parent = self

    def is_wildcard(self):
        return self.label == '*'

    def delete_child(self, child):
        label = child.label if isinstance(child, _Trie) else child
        assert label in self.children, "No child with label %s exists." % label
        del self.children[label]

    def compute_kgrams(self, X, k):
        for index in range(len(X)):
            self.kgrams[index] = np.array(
                [(offset, 0) for offset in range(len(X[index]) - k + 1)])

    def _process_node(self, X, k, m,  wildcard=False, mismatch=False):
        assert (wildcard or mismatch)
        if X.ndim == 1:
            XX = []
            for i in range(X.shape[0]):
                XX.append(np.array(list(X[i])))
            X = 1 * XX

        if wildcard:
            valsup = 0
        elif mismatch:
            valsup = m
        
        if self.is_root():
            self.compute_kgrams(X, k)

        elif self.is_wildcard():
            self.nb_wildcard += 1
        
        else:
            for index, substring_pointers in self.kgrams.items():
                substring_pointers[:, 1] += (
                    X[index][substring_pointers[..., 0] + self.level - 1] !=
                    self.label)
                self.kgrams[index] = np.delete(
                    substring_pointers,
                    np.nonzero(substring_pointers[..., 1] > valsup),
                    axis=0)

            self.kgrams = {
                index: substring_pointers
                for (index, substring_pointers) in self.kgrams.items()
                if len(substring_pointers) > 0
            }

        if wildcard:
            alive = (not self.is_empty()) and (self.nb_wildcard <= m)
        elif mismatch:
            alive = (not self.is_empty())

        return alive

    def _dfs(self,
             X,
             k=2,
             m=1,
             kernel=None,
             show=1,
             name=None,
             wildcard=False,
             mismatch=False):
        assert (wildcard or mismatch)
        length = len(self.vocab)
        if kernel is None:
            kernel = np.zeros((len(X), len(X)))
        n_kmers = 0
        alive = self.process_node(X, k, m)
        if alive:
            if k == 0:
                n_kmers += 1
                self.update_kernel(kernel)
            else:
                if show > 0:
                    if name is not None:
                        desc = "Trie DFS for " + name
                    else:
                        desc = "Trie DFS"
                    erange = self.vrange(length, desc=desc)
                else:
                    erange = range(length)
                for j in erange:
                    
                    Logger.indent()
                    if wildcard:
                        child = WildcardTrie(
                            la=self.la, label=self.vocab[j], parent=self)
                    elif mismatch:
                        child = MismatchTrie(label=self.vocab[j], parent=self)
                    kernel, child_kmers, child_alive = child.dfs(
                        X, k - 1, m, kernel=kernel, show=show-1)
                    if child.is_empty():
                        self.delete_child(child)
                    n_kmers += child_kmers if child_alive else 0
                    
                    Logger.dindent()
                    
        return kernel, n_kmers, alive


def normalized_kernel(K):
    K = 1 * K
    for i in range(K.shape[0]):
        for j in range(i + 1, K.shape[0]):
            q = np.sqrt(K[i, i] * K[j, j])
            if q > 0:
                K[i, j] /= q
                K[j, i] = K[i, j]
    np.fill_diagonal(K, 1.)
    return K + 1

class GappyTrie(_Trie) :
    def __init__(self, la = 1, label = None, parent = None, verbose = True):
        self.label = label
        self.level = 0
        self.la = la
        self.children = {}
        self.kgrams = {}
        self.parent = parent
        self.full_label = ''
        self.vocab = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
        if parent is not None:
            self.full_label = parent.full_label + self.label
            parent.add_child(self)
        self.verbose = verbose
        
    def is_root(self):
        return self.parent is None
    
    def update_kernel(self, kernel):
        for i in self.kgrams.keys():
            for j in self.kgrams.keys():
                kernel[i, j] += (len(self.kgrams[i])) * (len(self.kgrams[j]))
    
    def compute_kgrams(self, X, g):
        for index in range(len(X)):
            self.kgrams[index] = np.array(
                [(offset, -1) for offset in range(len(X[index]) - g + 1)])
    
    def process_node(self, X, g, l):
#        if X.ndim == 1:
#            XX = []
#            for i in range(X.shape[0]):
#                XX.append(np.array(list(X[i])))
#            X = 1 * XX
        
        if self.is_root():
            self.compute_kgrams(X, g)

        else:
            for index, substring_pointers in self.kgrams.items():
                for j in range(len(substring_pointers)) :
                    in_remainder = (X[index][substring_pointers[j,0] + substring_pointers[j,1] + 1: substring_pointers[j,0]  + g]).find(self.label)
                    if in_remainder < 0 :
                        substring_pointers[j,1] = g + 1
                    else : 
                        substring_pointers[j,1] = substring_pointers[j,1] + 1 + in_remainder
                    self.kgrams[index] = np.delete(
                            substring_pointers,
                            np.nonzero(substring_pointers[..., 1] > g),
                            axis=0)

            self.kgrams = {
                index: substring_pointers
                for (index, substring_pointers) in self.kgrams.items()
                if len(substring_pointers) > 0
            }

        alive = (not self.is_empty())

        return alive
    
    def dfs(self,
             X,
             g=2,
             l=1,
             kernel=None,
             show=1,
             name=None):
        length = len(self.vocab)
        if kernel is None:
            kernel = np.zeros((len(X), len(X)))
        n_kmers = 0
        alive = self.process_node(X, g, l)
        if alive:
            if l == 0:
                n_kmers += 1
                self.update_kernel(kernel)
            else:
                if show > 0:
                    if name is not None:
                        desc = "Trie DFS for " + name
                    else:
                        desc = "Trie DFS"
                    erange = self.vrange(length, desc=desc)
                else:
                    erange = range(length)
                for j in erange:
                    Logger.indent()
                    child = GappyTrie(label=self.vocab[j], parent=self)
                    kernel, child_kmers, child_alive = child.dfs(
                        X, g, l - 1, kernel=kernel, show=show-1)
                    if child.is_empty():
                        self.delete_child(child)
                    n_kmers += child_kmers if child_alive else 0
                    
                    Logger.dindent()
                    
        return kernel, n_kmers, alive

class WildcardTrie(_Trie):
    def __init__(self, la=1, label=None, parent=None, verbose=True):
        self.label = label
        self.level = 0
        self.children = {}
        self.kgrams = {}
        self.parent = parent
        self.la = la
        self.nb_wildcard = 0
        self.full_label = ''
        self.vocab = {0: 'A', 1: 'T', 2: 'G', 3: 'C', 4: '*'}
        if parent is not None:
            self.full_label = parent.full_label + self.label
            self.nb_wildcard = self.parent.nb_wildcard
            parent.add_child(self)

        self.verbose = verbose

    def is_root(self):
        return self.parent is None

    def process_node(self, X, k, m):
        return self._process_node(X, k, m, wildcard=True)

    def update_kernel(self, kernel):
        for i in self.kgrams.keys():
            for j in self.kgrams.keys():
                kernel[i, j] += (len(self.kgrams[i]) * len(self.kgrams[j])) * (
                    self.la**self.nb_wildcard)
        
    def dfs(self, X, k=2, m=1, kernel=None, show=1, name=None):
        return self._dfs(X, k, m, kernel, show, name=name, wildcard=True)


class MismatchTrie(_Trie):
    def __init__(self, la=1, label=None, parent=None, verbose=True):
        assert(la == 1)
        self.la = 1
        self.label = label
        self.level = 0
        self.children = {}
        self.kgrams = {}
        self.parent = parent
        self.full_label = ''
        self.vocab = {0: 'A', 1: 'T', 2: 'G', 3: 'C'}
        if parent is not None:
            self.full_label = parent.full_label + self.label
            parent.add_child(self)

        self.verbose = verbose

    def is_root(self):
        return self.parent is None

    def process_node(self, X, k, m):
        return self._process_node(X, k, m, mismatch=True)

    def update_kernel(self, kernel):
        for i in self.kgrams.keys():
            for j in self.kgrams.keys():
                kernel[i, j] += (len(self.kgrams[i]) * len(self.kgrams[j]))

    def dfs(self, X, k=2, m=1, kernel=None, show=1, name=None):
        return self._dfs(X, k, m, kernel, show, name=name, mismatch=True)


if __name__ == '__main__':
    data = np.array(['ATTA', 'AAAA'])
    import time
    debut = time.perf_counter()
    t = GappyTrie()
    ker, n_kmers, _ = t.dfs(data, 2, 1)
    # ipdb.set_trace()
    fin = time.perf_counter()
    print('temps = ', fin - debut)

    print(ker)
