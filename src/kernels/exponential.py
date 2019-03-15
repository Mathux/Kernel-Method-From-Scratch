from src.kernels.kernel import DataKernel, KernelCreate
import numpy as np


class ExponentialKernel(DataKernel, metaclass=KernelCreate):
    name = "exp"
    defaultParameters = {"sigma": 1}

    def kernel(self, x, y):
        return np.exp(-np.linalg.norm(x - y) / (2 * self.param.sigma**2))


if __name__ == "__main__":
    from src.tools.test import EasyTest
    EasyTest(kernel="exp", data="synth")
