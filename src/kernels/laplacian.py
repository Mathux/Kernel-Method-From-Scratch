from src.kernels.kernel import DataKernel, KernelCreate
import numpy as np


class LaplacianKernel(DataKernel, metaclass=KernelCreate):
    name = "laplacian"
    defaultParameters = {"sigma": 1}

    def kernel(self, x, y):
        return np.exp(-np.linalg.norm(x-y)/self.param.sigma)


if __name__ == "__main__":
    from src.tools.test import EasyTest
    EasyTest(kernel="laplacian", data="synth")
