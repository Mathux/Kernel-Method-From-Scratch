import numpy as np
from src.kernels.kernel import Kernel, KernelCreate


class GaussianKernel(Kernel, metaclass=KernelCreate):
    defaultParameters = {"sigma": 1}

    def kernel(self, x, y):
        return np.exp(-np.linalg.norm(x - y)**2) / self.param.sigma**2


if __name__ == "__main__":
    from src.data.synthetic import GenClassData
    data = GenClassData(100, 3, nclasses=2, mode="gauss")
    kernel = GaussianKernel(data)
    K = kernel.K
