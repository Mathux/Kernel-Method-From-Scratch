from src.kernels.kernel import DataKernel, KernelCreate


class PolynomialKernel(DataKernel, metaclass=KernelCreate):
    name = "poly"
    defaultParameters = {"offset": 10, "dim": 5}

    def kernel(self, x, y):
        return (self.param.offset + x.dot(y))**self.param.dim


if __name__ == "__main__":
    from src.tools.test import EasyTest
    EasyTest(kernel="poly", data="synth")
