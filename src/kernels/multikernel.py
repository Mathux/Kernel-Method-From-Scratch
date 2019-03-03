import numpy as np
from src.kernels.kernel import DataKernel, KernelCreate


def multicreate(classcreator):
    def f(dataset, kernels, parameters=None):
        return classcreator(
            dataset, parameters={
                "kernels": kernels,
                "_parameters": parameters
            })

    return f


@multicreate
class MultiKernel(DataKernel, metaclass=KernelCreate):
    defaultParameters = {"kernels": [], "_parameters": None}
    toreset = ["_KC", "_n", "_m", "_weights"]

    @property
    def weights(self):
        assert (self._weights is not None)
        return self._weights

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
        self.kernels = np.array(
            [k(value, param) for (k, param) in zip(kernels, kparams)])
        self.Ks = np.array([k.K for k in self.kernels])

    def __getitem__(self, key):
        assert (key in range(self.size))
        return self.Ks[key]
