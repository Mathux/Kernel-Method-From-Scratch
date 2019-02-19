import nose
import nose.tools
import numpy.testing
import numpy as np


def normalize_kernel(kernel):

    nkernel = numpy.copy(kernel)

    assert nkernel.ndim == 2
    assert nkernel.shape[0] == nkernel.shape[1]

    for i in xrange(nkernel.shape[0]):
        for j in xrange(i + 1, nkernel.shape[0]):
            q = np.sqrt(nkernel[i, i] * nkernel[j, j])
            if q > 0:
                nkernel[i, j] /= q
                nkernel[j, i] = nkernel[i, j]  # symmetry

    # finally, set diagonal elements to 1
    np.fill_diagonal(nkernel, 1.)

    return nkernel


class Trie(object):

    def __init__(self, label=None, parent=None):

        self.label = label  
        self.level = 0  
        self.children = {}
        self.kgrams = {}
        self.parent = parent
        if not parent is None:
            parent.add_child(self)

    def is_root(self):

        return self.parent is None

    def is_leaf(self):

        return len(self.children) == 0

    def is_empty(self):

        return len(self.kgrams) == 0

    def copy_kgrams(self):
  
        return {index: np.array(substring_pointers)
                for index, substring_pointers in self.kgrams.items()}

    def add_child(self, child):

        assert not child.label in self.children
        child.kgrams = self.copy_kgrams()
        child.level = self.level + 1
        self.children[child.label] = child
        child.parent = self

    def delete_child(self, child):

        label = child.label if isinstance(child, Trie) else child
        assert label in self.children, "No child with label %s exists." % label
        del self.children[label]

    def compute_kgrams(self, X, k):

        if not isinstance(X, np.ndarray):
            X = np.array(X)

        if X.ndim == 1:
            X = np.array([X])

        assert X.ndim == 2

        for index in range(len(X)):
            self.kgrams[index] = np.array([(offset,0) for offset in range(len(X[index]) - k + 1)])

    def process_node(self, X, k, m):

        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if X.ndim == 1:
            X = np.array([X])

        assert X.ndim == 2

        if self.is_root():
            self.compute_kgrams(X, k)
        else:
            for index, substring_pointers in self.kgrams.items():
                substring_pointers[..., 1] += (X[index][
                        substring_pointers[..., 0] + self.level - 1
                        ] != self.label)
                self.kgrams[index] = np.delete(substring_pointers,
                                               np.nonzero(
                        substring_pointers[..., 1] > m),
                                               axis=0)
            self.kgrams = {index: substring_pointers for (
                    index, substring_pointers) in self.kgrams.items(
                    ) if len(substring_pointers)}

        return not self.is_empty()

    def update_kernel(self, kernel, m) :
        for i in self.kgrams:
            for j in self.kgrams:
                kernel[i, j] += len(self.kgrams[i]) * len(self.kgrams[j])

    def dfs(self, X, k, m, kernel=None) :
        l = 4
        if kernel is None:
            kernel = np.zeros((len(X), len(X)))
        n_kmers = 0
        alive = self.process_node(X, k, m)
        if alive:
            if k == 0:
                n_kmers += 1
                self.update_kernel(kernel, m)
            else:
                for j in range(l):
                    child = Trie(label=j, parent=self)
                    kernel, child_kmers, child_alive = child.dfs(X, k - 1, m, kernel=kernel)
                    if child.is_empty():
                        self.delete_child(child)
                    n_kmers += child_kmers if child_alive else 0

        return kernel, n_kmers, alive


