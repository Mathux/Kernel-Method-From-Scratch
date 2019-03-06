import numpy as np
from src.tools.utils import Parameters, Logger
import pickle


class KernelCreate(type):
    def __init__(cls, clsname, superclasses, attributedict):
        if "toreset" not in cls.__dict__:
            cls.toreset = superclasses[0].toreset

        def init(self, dataset=None, parameters=None, verbose=True):
            super(cls, self).__init__(
                dataset=dataset,
                name=clsname,
                parameters=parameters,
                verbose=verbose,
                cls=cls)

        cls.__init__ = init


class Kernel(Logger):
    def __init__(self,
                 dataset=None,
                 name="Kernel",
                 parameters=None,
                 verbose=True,
                 toreset=["_K", "_KC", "_n", "_m"],
                 cls=None):
        self.verbose = verbose

        self._toreset = toreset

        self.param = Parameters(parameters, cls.defaultParameters)

        if dataset is not None:
            self.dataset = dataset
        else:
            self.reset()

        self.__name__ = name

    def __call__(self, x, y):
        return self.kernel(x, y)

    def reset(self):
        dic = self.__dict__
        for el in self._toreset:
            dic[el] = None

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
    
    def save_K(self) :
        pass
        

    @property
    def KC(self):
        # Check if KC is computed before
        if self._KC is None:
            # Compute the centered gram matrix
            self._compute_centered_gram()
        return self._KC
    
    def _compute_gram(self):
        K = np.zeros((self.n, self.n))
        for i in self.vrange(self.n, "Gram matrix of " + self.__name__):
            for j in range(i, self.n):
                K[i, j] = K[j, i] = self.kernel(self.data[i], self.data[j])
        self._K = K

    def _compute_centered_gram(self):
        K = self.K
        self._log("Center the gram matrix..")
        n = self.n
        oneN = (1 / n) * np.ones((n, n))
        self._KC = K - oneN.dot(K) - K.dot(oneN) + oneN.dot(K.dot(oneN))
        self._log("Gram matrix centered!")

    def kernel(self, x, y):
        return self._kernel(x, y) + 1
    
    def predict(self, x):
        return np.array([self.kernel(xi, x) for xi in self.data])

    def addconfig(self, name, value):
        self.config[name] = value

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__


class GenKernel(Kernel):
    def __init__(self,
                 dataset=None,
                 name="GenKernel",
                 parameters=None,
                 verbose=True,
                 cls=None):
        assert ("toreset" in cls.__dict__)
        toreset = cls.toreset
        super(GenKernel, self).__init__(
            dataset=dataset,
            name=name,
            parameters=parameters,
            verbose=verbose,
            toreset=toreset,
            cls=cls)


class DataKernel(GenKernel):
    toreset = ["_K", "_KC", "_n", "_m"]

    def __init__(self,
                 dataset=None,
                 name="DataKernel",
                 parameters=None,
                 verbose=True,
                 cls=None):
        super(DataKernel, self).__init__(
            dataset=dataset,
            name=name,
            parameters=parameters,
            verbose=verbose,
            cls=cls)


class StringKernel(GenKernel):
    toreset = ["_K", "_KC", "_n", "_m", "_phis", "_mers"]

    def __init__(self,
                 dataset=None,
                 name="StringKernel",
                 parameters=None,
                 verbose=True,
                 cls=None):
        super(StringKernel, self).__init__(
            dataset=dataset,
            name=name,
            parameters=parameters,
            verbose=verbose,
            cls=cls)

    def _compute_phis(self):
        phis = []
        for x in self.viterator(self.data, "Phis of " + self.__name__):
            phi = self._compute_phi(x)
            phis.append(phi)
        self._phis = np.array(phis)

    def predict(self, x):
        phix = self._compute_phi(x)
        return np.array([np.dot(phix, phi) for phi in self.phis])

    def _compute_gram(self):
        K = np.zeros((self.n, self.n))
        phis = self.phis

        for i in self.vrange(self.n, desc="Gram matrix of " + self.__name__):
            for j in range(i, self.n):
                K[i, j] = K[j, i] = np.dot(phis[i], phis[j])

        self._normalized_kernel(K)

    # Normalise the kernel (divide by the variance)
    def _normalized_kernel(self, K):
        for i in self.vrange(K.shape[0], "Normalise kernel"):
            for j in range(i + 1, K.shape[0]):
                q = np.sqrt(K[i, i] * K[j, j])
                if q > 0:
                    K[i, j] /= q
                    K[j, i] = K[i, j]
        np.fill_diagonal(K, 1.)
        self._K = K

    def kernel(self, x, y):
        return np.dot(self._compute_phi(x), self._compute_phi(y)) + 1

    @property
    def phis(self):
        # Check if phis are computed before
        if self._phis is None:
            # Compute phis
            self._compute_phis()
        return self._phis

    @property
    def mers(self):
        from itertools import product
        if self._mers is None:
            self._mers = [(''.join(c))
                          for c in product('ACGT', repeat=self.param.k)]
        return self._mers


def AllStringKernels():
    from src.kernels.spectral import SpectralKernel
    from src.kernels.mismatch import MismatchKernel
    from src.kernels.wd import WDKernel
    from src.kernels.la import LAKernel

    kernels = [MismatchKernel, SpectralKernel, WDKernel, LAKernel]
    names = ["mismatch", "spectral", "wd", "la"]
    return kernels, names


def AllDataKernels():
    from src.kernels.exponential import ExponentialKernel
    from src.kernels.gaussian import GaussianKernel
    from src.kernels.laplacian import LaplacianKernel
    from src.kernels.linear import LinearKernel
    from src.kernels.polynomial import PolynomialKernel
    from src.kernels.quad import QuadKernel
    from src.kernels.sigmoid import SigmoidKernel

    kernels = [
        ExponentialKernel, GaussianKernel, LaplacianKernel, LinearKernel,
        PolynomialKernel, QuadKernel, SigmoidKernel
    ]
    names = [
        "exp", "gaussian", "laplacian", "linear", "poly", "quad", "sigmoid"
    ]
    return kernels, names


def AllKernels():
    k1, n1 = AllStringKernels()
    k2, n2 = AllDataKernels()
    return k1 + k2, n1 + n2
