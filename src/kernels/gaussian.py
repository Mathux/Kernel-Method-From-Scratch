import numpy as np
from src.kernels.kernel import DataKernel, KernelCreate


class GaussianKernel(DataKernel, metaclass=KernelCreate):
    name = "gaussian"
    defaultParameters = {"sigma": 1}

    def kernel(self, x, y):
        return np.exp(-np.linalg.norm(x - y)**2) / (2*self.param.sigma**2)


if __name__ == "__main__":
    from src.tools.test import EasyTest
    EasyTest(kernels="gaussian", data="synth")
