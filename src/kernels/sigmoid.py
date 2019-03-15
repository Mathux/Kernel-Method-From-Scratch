from src.kernels.kernel import DataKernel, KernelCreate
from numpy import tanh


class SigmoidKernel(DataKernel, metaclass=KernelCreate):
    name = "sigmoid"
    defaultParameters = {"offset": 30, "alpha": 10}

    def kernel(self, x, y):
        return tanh(self.param.alpha * x.dot(y) + self.param.offset)


if __name__ == "__main__":
    from src.tools.test import EasyTest
    EasyTest(kernel="sigmoid", data="synth")
