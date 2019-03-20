import numpy as np
from src.kernels.kernel import GenKernel, KernelCreate
from src.kernels.kernel import AllKernels

kernels, kernelsNames = AllKernels(multi=False)


def find(name, allS):
    return allS.index(name)


def findKernel(name):
    return kernels[find(name, kernelsNames)]


class MultiKernel(GenKernel, metaclass=KernelCreate):
    name = "multikernel"
    defaultParameters = {"kernels": [], "_parameters": None, "_weights": None}
    toreset = ["_KC", "_n", "_m", "_weights"]

    @property
    def weights(self):
        if self._weights is None:
            self._weights = self.param._weights
        return self._weights

    def predict(self, x):
        return np.sum([k.predict(x) for k in self.kernels], axis=0)
    
    @weights.setter
    def weights(self, value):
        self._weights = value

    @property
    def K(self):
        return self.Kw(self.weights)

    # Combinaison of kernels
    def Kw(self, weights):
        return np.einsum("ijk,i->jk", self.Ks, weights)

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

        kernels = self.param.kernels
        self.size = len(kernels)

        kparams = self.param._parameters
        if kparams is None:
            kparams = [None] * self.size
        self.kernels = np.array([
            findKernel(k)(value, param)
            for (k, param) in zip(kernels, kparams)
        ])
        self.Ks = np.array([k.K for k in self.kernels])

    def __getitem__(self, key):
        assert (key in range(self.size))
        return self.Ks[key]


if __name__ == "__main__":
    dparams = {"small": True, "nsmall": 100}
    kparams = {"k": 6, "m": 0, "trie": True}
    parameters = {
        "kernels": ["spectral", "wildcard"],
        "_parameters": [{
            "k": 6,
            "trie": True
        }, {
            "k": 5,
            'm': 1,
            'la': 1,
            "trie": True
        }],
        "_weights": [0.5, 0.5]
    }
    from src.tools.test import EasyTest

    EasyTest(
        kernels="multikernel",
        data="seq",
        dparams=dparams,
        kparams=parameters)
