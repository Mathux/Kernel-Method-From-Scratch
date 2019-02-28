import numpy as np
from src.tools.utils import Logger


class Kernel(Logger):
    def __init__(self,
                 dataset=None,
                 name="NONAME",
                 isString=False,
                 verbose=True):
        self.verbose = verbose
        if dataset is not None:
            self.dataset = dataset
        else:
            self.reset()
        self.config = {}
        self.isString = isString
        self.__name__ = name

    def reset(self):
        self._K = None
        self._KC = None
        self._n = None
        self._m = None
        self._phis = None
        
    @property
    def data(self):
        if self._data is None:
            raise ValueError("This kernel didn't have any data")
        return self._data

    @property
    def labels(self):
        if self._labels is None:
            raise ValueError("This kernel didn't have any labels")
        return self._labels

    @property
    def dataset(self):
        if self._dataset is None:
            raise ValueError("This kernel didn't have any dataset")
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value
        self._data = value.data
        self._labels = value.labels
        self.reset()
        # Init some values
        self._n = value.data.shape[0]
        self._m = value.data.shape[1] if len(value.data.shape) > 1 else None

    @property
    def n(self):
        if self._n is None:
            raise ValueError("This kernel didn't have any data, so no shape")
        return self._n

    @property
    def m(self):
        if self._m is None:
            raise ValueError("This kernel didn't have any data, so no shape")
        return self._m

    @property
    def K(self):
        # Check if K is computed before
        if self._K is None:
            # Compute the gram matrix
            self._compute_gram()
        return self._K

    @property
    def KC(self):
        # Check if KC is computed before
        if self._KC is None:
            # Compute the centered gram matrix
            self._compute_centered_gram()
        return self._K

    def _compute_gram(self):
        self._log("Computing the gram matrix..")
        K = np.zeros((self.n, self.n))
        for i in self.vrange(self.n):
            for j in range(i, self.n):
                K[i, j] = K[j, i] = self.kernel(self.data[i], self.data[j])
        self._K = K
        self._log("Gram matrix computed!")

    def _compute_centered_gram(self):
        K = self.K
        self._log("Center the gram matrix..")
        n = self.n
        oneN = (1 / n) * np.ones((n, n))
        self._KC = K - oneN.dot(K) - K.dot(oneN) + oneN.dot(K.dot(oneN))
        self._log("Gram matrix centered!")

    def predict(self, x):
        return np.array([self.kernel(xi, x) for xi in self.data])

    def addconfig(self, name, value):
        self.config[name] = value
    

class StringKernel(Kernel):
    def __init__(self, dataset=None, name="string", verbose=True):
        super(StringKernel, self).__init__(
            dataset=dataset, name=name, isString=True, verbose=verbose)
        self._phis = None

    def _compute_phis(self):
        self._log("Computing phis..")
        phis = []
        for x in self.viterator(self.data):
            phi = self._compute_phi(x)
            phis.append(phi)
        self._phis = np.array(phis)
        self._log("phis computed!")

    def predict(self, x):
        phix = self._compute_phi(x)
        return np.array([np.dot(phix, phi) for phi in self.phis])

    def _compute_gram(self):
        K = np.zeros((self.n, self.n))
        phis = self.phis

        self._log("Computing the gram matrix..")
        for i in self.vrange(self.n):
            for j in range(i, self.n):
                K[i, j] = K[j, i] = np.dot(phis[i], phis[j])

        self._log("Gram matrix computed!")
        self._normalized_kernel(K)

    # Normalise the kernel (divide by the variance)
    def _normalized_kernel(self, K):
        self._log("Normalise the kernel..")
        for i in self.vrange(K.shape[0]):
            for j in range(i + 1, K.shape[0]):
                q = np.sqrt(K[i, i] * K[j, j])
                if q > 0:
                    K[i, j] /= q
                    K[j, i] = K[i, j]
        np.fill_diagonal(K, 1.)
        self._log("Kernel normalized!")
        self._K = K
        
    def kernel(self, x, y):
        return np.dot(self._compute_phi(x), self._compute_phi(y))

    @property
    def phis(self):
        # Check if phis are computed before
        if self._phis is None:
            # Compute phis
            self._compute_phis()
        return self._phis
